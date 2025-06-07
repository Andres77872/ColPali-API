INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "operation_type": {
            "type": "string",
            "enum": ["embed_images", "embed_query"],
            "description": "Type of operation to perform"
        },
        "images": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of image URLs or base64 encoded images (for embed_images operation)"
        },
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of query strings (for embed_query operation)"
        },
        "size": {
            "type": "integer",
            "default": 3584,
            "description": "Image resize height"
        },
        "pool_factor": {
            "type": "integer",
            "default": 2,
            "description": "Token pooling factor"
        }
    },
    "required": ["operation_type"]
}
