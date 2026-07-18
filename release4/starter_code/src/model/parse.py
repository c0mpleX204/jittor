from .spec import ModelSpec
from .dcvm import DirectionDistanceVelocityModule
from .distance_vm import StraightPCFDistanceVelocityModule
from .vm import VelocityModule

def get_model(model_config, **kwargs) -> ModelSpec:
    MAP = {
        'DirectionDistanceVelocityModule': DirectionDistanceVelocityModule,
        'StraightPCFDistanceVelocityModule': StraightPCFDistanceVelocityModule,
        'VelocityModule': VelocityModule,
    }
    __target__ = model_config['__target__']
    del model_config['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    return MAP[__target__](model_config=model_config, **kwargs)
