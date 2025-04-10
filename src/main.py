import asyncio
from viam.module.module import Module
try:
    from models.siamese_network_vision import SiameseNetworkVision
except ModuleNotFoundError:
    # when running as local module with run.sh
    from .models.siamese_network_vision import SiameseNetworkVision


if __name__ == '__main__':
    asyncio.run(Module.run_from_registry())
