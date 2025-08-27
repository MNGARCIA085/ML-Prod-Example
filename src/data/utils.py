from .preprocessor import BreastCancerPreprocessor
from .preprocessor_with_normalization import BreastCancerPreprocessorNormalized



# get a single preprocessor
def get_preprocessor(name, batch_size):
    """
    Return a preprocessor instance for the given data variant.

    Args:
        name (str): The name of the data variant. 
            Supported values: "simple", "standardize".
        batch_size (int): Batch size to use in the preprocessor pipeline.

    Returns:
        BreastCancerPreprocessor or BreastCancerPreprocessorNormalized: 
        An initialized preprocessor corresponding to the chosen variant.

    Raises:
        ValueError: If an unsupported data variant name is provided.
    """
    mapping = {
        "simple": BreastCancerPreprocessor(batch_size=batch_size),
        "standardize": BreastCancerPreprocessorNormalized(batch_size=batch_size),
    }
    if name not in mapping:
        raise ValueError(f"Unknown data variant: {name}")
    return mapping[name]



# preprocessors
def get_preprocessors(data_variants, batch_size):
    """
    Return a dictionary of preprocessors for the given data variants.

    Args:
        data_variants (list[str]): List of variant names to build.
            Each element must be either "simple" or "standardize".
        batch_size (int): Batch size to use in each preprocessor pipeline.

    Returns:
        dict[str, object]: A mapping from variant name to its corresponding 
        preprocessor instance.
    """
    preprocessors = {
        name: get_preprocessor(name, batch_size) for name in data_variants
    }
    return preprocessors