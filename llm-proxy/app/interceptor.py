import hashlib
import json
import httpx
import logging
import structlog
from typing import Any, Dict, List, Union, Optional
from abc import ABC, abstractmethod
import re

from presidio_analyzer import AnalyzerEngine, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_anonymizer.operators import Operator, OperatorType
#from presidio_anonymizer.models import OperatorType
from presidio_analyzer.nlp_engine import NlpEngineProvider

from app.regex_rules import rules as rr
gfilter = None
gfilter2 = None
logger = structlog.get_logger()
nlp_configuration = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "en_core_web_lg"},   # English
        {"lang_code": "uk", "model_name": "uk_core_news_lg"},  # Ukrainian
        {"lang_code": "ru", "model_name": "ru_core_news_lg"}   # Russian (Added)
    ],
}

class InterceptorError(Exception):
    pass

def intercept_request(interceptor_type: str, data: List[Dict[str, Any]], filter_config: Optional[Any] = None) -> Union[Dict[str, Any], str]:
    """
    Intercepts and processes LLM requests (list of messages) based on the provided interceptor type.
    """
    global gfilter
    global gfilter2
    try:
        if interceptor_type == "log":
            # Basic logging interception
            return {"status": "success", "processed_data": data, "message": "Request logged"}
        elif interceptor_type == "filter":
            if filter_config and filter_config.enabled:
                if gfilter:
                    filter = gfilter
                else:
                    if filter_config.type == "regexp":
                        
                        rules = rr
                        filter = RegExpFilter(rules)
                    elif filter_config.type == "llm":
                        filter = LLMFilter(url=filter_config.llm_url, prompt=filter_config.llm_prompt)
                    elif filter_config.type == "presidio":
                        filter = PresidioFilter()
                    else:
                        raise InterceptorError(f"Unknown filter type: {filter_config.type}")
                    gfilter = filter
                processed_data = filter.process(data)
                if filter_config.regexphybrid:
                    if not gfilter2:
                        rules = rr
                        gfilter2 = RegExpFilter(rules)
                    processed_data = gfilter2.process(processed_data)

                return {"status": "success", "processed_data": processed_data, "message": f"Request filtered with {filter_config.type}"}
            
            # Default to logging only or pass-through if filter disabled
            return {"status": "success", "processed_data": data, "message": "No filter active"}
        else:
            raise InterceptorError(f"Unknown interceptor type: {interceptor_type}")
    except Exception as e:
        return {"status": "error", "message": str(e)}
# elif interceptor_type == "filter":
#             # Basic filtering: remove messages with 'role' as 'system'
#             # processed_data = [msg for msg in data if msg.get("role") != "system"]
#             # return {"status": "success", "processed_data": processed_data, "message": "Request filtered"}
#             rules = rr
#             filter = RegExpFilter(rules)
#             # filter = PresidioFilter()
#             # return  {"status": "success", "processed_data": filter.process(data), "message": "Request filtered with regexp"}
#             # Decide which filter to use
#             # if llm_filter_config and llm_filter_config.enabled:
#             #     filter = LLMFilter(url=llm_filter_config.url, prompt=llm_filter_config.prompt)
#             # else:
#             #     filter = PresidioFilter()

#             return  {"status": "success", "processed_data": filter.process(data), "message": "Request filtered"}
#         else:
#             raise InterceptorError(f"Unknown interceptor type: {interceptor_type}")
#     except Exception as e:
#         return {"status": "error", "message": str(e)}

class BaseFilter(ABC):
    def __init__(self, cache = None):
        self.cache = cache

    def _generate_cache_key(self, data: List[Dict[str, Any]]) -> str:
        # Create a unique key based on the content of the data
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.cache:
            cache_key = self._generate_cache_key(data)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result

        # Run the specific filter logic
        processed_data = self._filter_logic(data)

        if self.cache:
            self.cache.set(cache_key, processed_data)

        return processed_data

    @abstractmethod
    def _filter_logic(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Specific filtering implementation."""
        pass

# Example of a concrete filter implementation
class SystemMessageFilter(BaseFilter):
    def _filter_logic(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [msg for msg in data if msg.get("role") != "assistant"]

class RegExpFilter(BaseFilter):
    def __init__(self, rules: List[Dict[str, str]], cache=None):
        """
        :param rules: List of dicts, e.g., [{"pattern": r"\d+", "replacement": "X"}]
        """
        super().__init__(cache)
        self.rules = [{"pattern": re.compile(r["pattern"]), "replacement": r["replacement"]} for r in rules]

    def _filter_logic(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed_data = []
        for msg in data:
            new_msg = msg.copy()
            if "content" in new_msg:
                for rule in self.rules:
                    new_msg["content"] = rule["pattern"].sub(rule["replacement"], new_msg["content"])
                
                # Replace double spaces with single ones
                new_msg["content"] = re.sub(r' +', ' ', new_msg["content"])
            processed_data.append(new_msg)
        return processed_data




class PresidioFilter(BaseFilter):
    def __init__(self,  cache=None):
        super().__init__(cache)
        # self.entities = entities
        provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
        nlp_engine = provider.create_engine()
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        self.anonymizer = AnonymizerEngine()
        
        # self.operators_config = {
        #                 "DEFAULT": OperatorConfig(
        #                 "custom", 
        #                 {"lambda": lambda text, params=None: f"[{params['entity_type']}]"}
        #                 )
        #                 }
        
    def make_bracket_multiplier(self, entity_type_name):
        def replacer(entity_text: str) -> str:
            # Split the text by whitespace to get the true token count
            token_count = len(entity_text.split())
            # Multiply the bracketed tag by the number of found tokens
            return " ".join([f"[{entity_type_name}]"] * token_count)
        return replacer

    def _filter_logic(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Prototype: In a real implementation, call Presidio analyzer/anonymizer here
        processed_data = []
        for msg in data:
            new_msg = msg.copy()
            if "content" in new_msg:
                analyzer_results = self.analyzer.analyze(text=new_msg["content"], 
                                language="uk")
                
                operators_mapping = {}
                for result in analyzer_results:
                    operators_mapping[result.entity_type] = OperatorConfig(
                        operator_name="custom",
                        params={"lambda": self.make_bracket_multiplier(result.entity_type)}
                    )

                anonymized_text = self.anonymizer.anonymize(text=new_msg["content"], 
                                analyzer_results=analyzer_results,
                                operators=operators_mapping
                                )
                print(anonymized_text)
                new_msg["content"] = anonymized_text.text
                # Placeholder: Masking demo for entities
                # for entity in self.entities:
                    
                #     new_msg["content"] = new_msg["content"].replace(f"<{entity}>", f"[{entity}_MASKED]")
            processed_data.append(new_msg)
        return processed_data

class LLMFilter(BaseFilter):
    def __init__(self, url: str, prompt: str, cache=None):
        super().__init__(cache)
        self.url = url
        self.prompt = prompt

    def _filter_logic(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed_data = []
        try:
            with httpx.Client(timeout=30.0) as client:
                for msg in data:
                    new_msg = msg.copy()
                    if "content" in new_msg:
                        try:

                            prt = """
                                  You are a data privacy utility. Replace all PII with tags like
                                   [NAME], [EMAIL], [ADDRESS], [BANK_ACCOUNT], [BANK_CARD_NUMBER], [PASSPORT], [LIC_PLATE], [RNOKPP], [UNZR].
                                   Every separate word should have it's separate tag.
                                   List of known patterns Ukrainian: 
                                   Registration Number of the Taxpayer's Account Card (РНОКПП / Formerly ІПН) RNOKPP: 10 digits XXXXXXXXXX;
                                   Unique Document Record Number (УНЗР) UNZR: YYYYMMDD-XXXXX (8 digits, a hyphen, and 5 digits);
                                   Passport number Two cyrillic letters and 6 digits for old one (НК123456) or 9 digits for new one XXXXXXXXX;
                                     Matches Ukrainian mobile and landline numbers starting with +380, 380, or domestic 0 format;
                                     Modern Ukrainian license plates generally feature a 2-letter region code, 4 digits, and a 2-letter suffix (e.g., AA 1234 BP or KA 9876 CE);
                                    Address can have different format but usually it's combination of street name and building number.
                                    NEVER join the words with "_" ( underscore symbol), all words should be separate.
                                   also pay attention to credit card numbers (including CVC codes and valid till date), email addresses and ip.
                                   Words that are not PII should be left as they are without any changes. 
                                   Use for Output JSON only.  
                            """
                            # Standard KoboldCPP format
                            payload = {
                                #"prompt": f"{self.prompt}\n\n{new_msg['content']}",
                                "prompt": f"<|im_start|>system\n{prt}<|im_end|>\n<|im_start|>user\nMask this: {new_msg['content']}<|im_end|>\n<|im_start|>assistant\n",
                                "max_length": 256,
                                "temperature": 0,
                                "grammar": "root ::= \"{\\\"masked_text\\\": \\\"\" string \"\\\"}\"\nstring ::= [^\"]*"
                                #"top_p": 0.9
                            }
                            response = client.post(self.url, json=payload)
                            if response.status_code == 200:
                                # KoboldCPP returns results in 'results' list, taking first
                                results = response.json().get("results", [])
                                #print(results)
                                if results:
                                    # Extrapolate text if needed or just use first result
                                    cnt= results[0].get("text", new_msg["content"])
                                    new_msg["content"] = json.loads(cnt)["masked_text"]
                            else:
                                logger.error(f"LLMFilter failed with status {response.status_code}")
                        except Exception as e:
                            logger.error(f"LLMFilter request error: {e}")
                    processed_data.append(new_msg)
                print(processed_data)    
        except Exception as e:
            logger.error(f"LLMFilter client initialization error: {e}")
            return data
        return processed_data
