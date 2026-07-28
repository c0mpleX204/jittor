from .spec import ModelSpec
from .cd_refine import (
    CDRefineModule,
    NoisyAnchorScoreFieldRefineModule,
    TangentialCDRefineModule,
)
from .dcvm import DirectionDistanceVelocityModule
from .straightpcf_vm_dm import (
    StraightPCFCoupledVelocityModule,
    StraightPCFVelocityDistanceModule,
)
from .vm import VelocityModule

def get_model(model_config, **kwargs) -> ModelSpec:
    MAP = {
        'CDRefineModule': CDRefineModule,
        'DirectionDistanceVelocityModule': DirectionDistanceVelocityModule,
        'NoisyAnchorScoreFieldRefineModule': NoisyAnchorScoreFieldRefineModule,
        'StraightPCFCoupledVelocityModule': StraightPCFCoupledVelocityModule,
        'StraightPCFVelocityDistanceModule': StraightPCFVelocityDistanceModule,
        'TangentialCDRefineModule': TangentialCDRefineModule,
        'VelocityModule': VelocityModule,
    }
    __target__ = model_config['__target__']
    del model_config['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    return MAP[__target__](model_config=model_config, **kwargs)
