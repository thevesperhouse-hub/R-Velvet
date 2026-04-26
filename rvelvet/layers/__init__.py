from .local_attention import LocalAttention
from .segment_compressor import SegmentCompressor, AdaptiveSegmentCompressor
from .global_reasoner import GlobalReasoner
from .memory_controller import MemoryController
from .adaptive_router import SegmentScanner, AdaptiveRouter
from .lora_adapter import LoRAAdapter, IterationLoRABank
from .halting import HaltingUnit, compute_halting_loss
from .iterative_reasoner import IterativeReasoner
