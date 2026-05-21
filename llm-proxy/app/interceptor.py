from typing import Any, Dict, List, Union, Optional
from abc import ABC, abstractmethod

class InterceptorError(Exception):
    pass

def intercept_request(interceptor_type: str, data: List[Dict[str, Any]]) -> Union[Dict[str, Any], str]:
    """
    Intercepts and processes LLM requests (list of messages) based on the provided interceptor type.
    """
    try:
        if interceptor_type == "log":
            # Basic logging interception
            return {"status": "success", "processed_data": data, "message": "Request logged"}
        elif interceptor_type == "filter":
            # Basic filtering: remove messages with 'role' as 'system'
            # processed_data = [msg for msg in data if msg.get("role") != "system"]
            # return {"status": "success", "processed_data": processed_data, "message": "Request filtered"}
            filter = SystemMessageFilter()
            return filter.process(data)
        else:
            raise InterceptorError(f"Unknown interceptor type: {interceptor_type}")
    except Exception as e:
        return {"status": "error", "message": str(e)}

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