rules = [
                {"pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "replacement": " [EMAIL_REDACTED] "},
                {"pattern": r"\d{3}-\d{3}-\d{4}", "replacement": " [PHONE_NUMBER] "},
                {"pattern": r"\+([1-9]\d{0,2})-([2-9]\d{2})-(\d{3})-(\d{4})\b", "replacement": " [PHONE_NUMBER] "},
                {"pattern": r"\b8-([9]\d{2})-(\d{3})-(\d{2})-(\d{2})\b", "replacement": " [PHONE_NUMBER] "}, #ru
                {"pattern": r"\b(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*\)|([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2}))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})\b", "replacement": "[PHONE_NUMBER]"},#US
                {"pattern":r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", "replacement":" [IBAN] "},
                {"pattern":r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+.[A-Za-z]{2,}\b", "replacement":" [EMAIL] "},
                {"pattern":r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b", "replacement":" [BANK_CARD_NUMBER] "},
                {"pattern":r"\b[4-6]\d{3}[ -]\d{4}[ -]\d{4}[ -]\d{4}\b", "replacement":" [BANK_CARD_NUMBER] [BANK_CARD_NUMBER] [BANK_CARD_NUMBER] [BANK_CARD_NUMBER] "},
                {"pattern":r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b", "replacement":" [IPV4] "},
                {"pattern": r"\b[A-ZА-Я]{2}\s*\d{4}\s*[A-ZА-Я]{2}\b", "replacement": " [VEHICLE_LICENSE] "},
                {"pattern": r"\b(?:\+?38)?(?:\s*\(?0\d{2}\)?)\s*\d{3}\s*[-]?\s*\d{2}\s*[-]?\s*\d{2}\b", "replacement": " [PHONE_NUMBER] "},
                {"pattern": r"\b(?:[А-ЩЬЮЯЄІЇҐ]{2}\d{6}|\d{9})\b", "replacement": " [PASSPORT_NUMBER] "},
                {"pattern": r"\b[A-Z0-9]{9}\b", "replacement": " [PASSPORT_NUMBER] "}, #US
                {"pattern": r"\b\d{8}-\d{5}\b", "replacement": " [UNZR] "},
                {"pattern": r"\b\d{10}\b", "replacement": " [RNOKPP] "},
                {"pattern": r"\b\d{12}\b", "replacement": " [INN] "},
                {"pattern": r"\b(\d{13})\b", "replacement": " [OGRN] "},
                {"pattern": r"\b(\d{11})\b", "replacement": " [SNILS] "},
                {"pattern": r"\b(\d{3})-(\d{3})-(\d{3})\s+(\d{2})\b", "replacement": " [SNILS] "},
                {"pattern": r"\b(\d{15})\b", "replacement": " [OGRNIP] "},
                
                {"pattern": r"\b(\d{2})[\s-]*(\d{2})[\s-]*(\d{6})\b", "replacement": " [PASSPORT_NUMBER] "}, #ru
                {"pattern": r"\b9\d{2}[- ]?(5\d|7\d|8\d|9[0-24-9])[- ]?\d{4}\b", "replacement": " [ITIN] "},
                {"pattern": r"\b(?!000|666|9\d{2})([0-8]\d{2}|7([0-6]\d|7[012]))([-]?)\d{2}\3\d{4}\b", "replacement": " [SSN] "},
                {"pattern": r"\b([A-Z]{2})\s+(\d{2})\s+(\d{2})\s+(\d{2})\s+([A-Z])\b", "replacement": " [SSN] [SSN] [SSN] [SSN] [SSN] "},
                {"pattern": r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?,\s+(Q[1-4])\s+(\d{4})\b", "replacement": " [DATE] "},
                {"pattern": r"\b(\d{4})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b", "replacement": " [DATE] "},
                {"pattern": r"\b(0?[1-9]|[12]\d|3[01])-/.-/.\b", "replacement": " [DATE] "},
                {"pattern": r"\b(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+(?:of\s+)?(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{4})\b", "replacement": " [DATE] "},
                {"pattern": r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?,?\s+(\d{4})\b", "replacement": " [DATE] "},
                {"pattern": r"(^|[^0-9\s]|\s(?!\d))(\d{3})(?!\s*\d|\d)", "replacement": " [CVC] "},

                ]