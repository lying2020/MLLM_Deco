from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig

# MPT is optional - only imported when needed (when model name contains 'mpt')
# This avoids import errors when MPT dependencies are not available
try:
    from .language_model.llava_mpt import LlavaMptForCausalLM, LlavaMptConfig
except ImportError:
    # MPT not available - will be imported on-demand in builder.py if needed
    LlavaMptForCausalLM = None
    LlavaMptConfig = None
