
class validated_number:
    def __init__(self, name=None, validator=None):
        self.name = name
        self.validator = validator

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, type=None) -> object:
        return obj.__dict__.get(self.name, 0)

    def __set__(self, obj, value) -> None:
        if self.validator:
            self.validator(value, self.name)
        obj.__dict__[self.name] = value

def validate_positive(value, name):
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive int or float.")

def validate_positive_t(value, name):
    if not isinstance(value, (int, float)) or value <= 1 / 10000: #must be some time left before expiration
        raise ValueError(f"{name} must be a positive int or float greater than 1/10000.")

def validate_r(value, name):
    if not isinstance(value, (int, float)) or not (-1.5 <= value <= 1.5): #define bounds for interest rates
        raise ValueError(f"{name} must be an int or float between -1.5 and 1.5.")

def validate_sigma(value, name):
    if not isinstance(value, (int, float)) or not (0 < value <= 4): #define upper bounds for volatility
        raise ValueError(f"{name} must be a positive int or float no greater than 4.")

def validate_q(value, name):
    if not isinstance(value, (int, float)) or not (0 <= value <= 1):
        raise ValueError(f"{name} must be a positive int or float between 0 and 1.")
