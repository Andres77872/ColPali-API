INPUT_SCHEMA = {
    "operation_type": {
        "type": str,
        'required': True,
        "default": None
    },
    "images": {
        "type": list,
        'required': False,
        "default": None
    },
    "queries": {
        "type": list,
        'required': False,
        "default": None
    },
    "size": {
        "type": int,
        'required': False,
        "default": 3584
    },
    "pool_factor": {
        "type": int,
        'required': False,
        "default": 2
    }
}
