from .spec import ModelSpec
from .cd_refine import (
    CDRefineModule,
    DirectionCorrectedNoisyGuidedRefineModule,
    LocalAttentionCDRefineModule,
    NoisyGuidedAlphaRefineModule,
    RiskAwareCDRefineModule,
    TangentSpreadRefineModule,
    TangentialCDRefineModule,
)
from .coupled_vm import CoupledSurfaceStraightVelocityModule
from .dcvm import DirectionDistanceVelocityModule
from .distance_vm import StraightPCFDistanceVelocityModule
from .edge_refine import EdgeRefineModule
from .straightpcf_vm_dm import (
    StraightPCFCoupledVelocityModule,
    StraightPCFVelocityDistanceModule,
)
from .vm import VelocityModule

def get_model(model_config, **kwargs) -> ModelSpec:
    MAP = {
        'CoupledSurfaceStraightVelocityModule': CoupledSurfaceStraightVelocityModule,
        'CDRefineModule': CDRefineModule,
        'DirectionCorrectedNoisyGuidedRefineModule': DirectionCorrectedNoisyGuidedRefineModule,
        'DirectionDistanceVelocityModule': DirectionDistanceVelocityModule,
        'EdgeRefineModule': EdgeRefineModule,
        'LocalAttentionCDRefineModule': LocalAttentionCDRefineModule,
        'NoisyGuidedAlphaRefineModule': NoisyGuidedAlphaRefineModule,
        'RiskAwareCDRefineModule': RiskAwareCDRefineModule,
        'TangentSpreadRefineModule': TangentSpreadRefineModule,
        'StraightPCFCoupledVelocityModule': StraightPCFCoupledVelocityModule,
        'StraightPCFVelocityDistanceModule': StraightPCFVelocityDistanceModule,
        'StraightPCFDistanceVelocityModule': StraightPCFDistanceVelocityModule,
        'TangentialCDRefineModule': TangentialCDRefineModule,
        'VelocityModule': VelocityModule,
    }
    __target__ = model_config['__target__']
    del model_config['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    return MAP[__target__](model_config=model_config, **kwargs)
