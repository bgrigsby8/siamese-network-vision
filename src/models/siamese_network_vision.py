import numpy as np
from PIL import Image, JpegImagePlugin
import tensorflow as tf
from typing import ClassVar, List, Mapping, Optional, Sequence

from typing_extensions import Self
from viam.components.camera import Camera
from viam.media.utils.pil import viam_to_pil_image
from viam.media.video import ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.vision import *
from viam.utils import ValueTypes

class SiameseNetworkVision(Vision, EasyResource):
    # To enable debug-level logging, either run viam-server with the --debug option,
    # or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(
        ModelFamily("brad-grigsby", "siamese-network"), "siamese-network-vision"
    )
    
    def __init__(self, name: str):
        super().__init__(name=name)
        self.model_name = ""
        self.camera = None

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """This method creates a new instance of this Vision service.
        The default implementation sets the name from the `config` parameter and then calls `reconfigure`.

        Args:
            config (ComponentConfig): The configuration for this resource
            dependencies (Mapping[ResourceName, ResourceBase]): The dependencies (both implicit and explicit)

        Returns:
            Self: The resource
        """
        return super().new(config, dependencies)

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Sequence[str]:
        """This method allows you to validate the configuration object received from the machine,
        as well as to return any implicit dependencies based on that `config`.

        Args:
            config (ComponentConfig): The configuration for this resource

        Returns:
            Sequence[str]: A list of implicit dependencies
        """
        model_path = config.attributes.fields["model_path"].string_value
        if model_path == "":
            raise ValueError("A model path is required for SiameseNetworkVision")
        
        reference_image_path = config.attributes.fields["reference_image_path"].string_value
        if reference_image_path == "":
            raise ValueError("A reference image path is required for SiameseNetworkVision")
        
        camera_name = config.attributes.fields["camera_name"].string_value
        if camera_name == "":
            raise ValueError("A camera name is required for SiameseNetworkVision")
        
        return [camera_name]

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """This method allows you to dynamically update your service when it receives a new `config` object.

        Args:
            config (ComponentConfig): The new configuration
            dependencies (Mapping[ResourceName, ResourceBase]): Any dependencies (both implicit and explicit)
        """
        self.camera_name = config.attributes.fields["camera_name"].string_value
        self.camera: Camera = dependencies[Camera.get_resource_name(self.camera_name)]
        self.model_path = config.attributes.fields["model_path"].string_value
        self.reference_image_path = config.attributes.fields["reference_image_path"].string_value
        self.tolerance = config.attributes.fields["tolerance"].number_value if "tolerance" in config.attributes.fields else 0.5

        return super().reconfigure(config, dependencies)
    
    def test_image_quality(self, test_image, threshold):
        """
        Test if an image is good or bad compared to a reference image using a Siamese network.
        
        Args:
            model_path: Path to the tfLite model
            test_image: PIL images to be tested
            threshold: Similarity threshold (lower values indicate more similarity)
        
        Returns:
            Tuple of (classification, similarity_score)
        """
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()

        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Load and preprocess images
        def preprocess_image(image):
            if not isinstance(image, (Image.Image, JpegImagePlugin.JpegImageFile)):
                img = Image.open(image).convert("RGB")
            else:
                img = image.convert("RGB")
            
            if len(input_details) > 0:
                input_shape = input_details[0]['shape']
                input_dtype = input_details[0]['dtype']
                if len(input_shape) == 4:
                    img = img.resize((input_shape[1], input_shape[2]))
                elif len(input_shape) == 3:
                    img = img.resize((input_shape[0], input_shape[1]))
                else:
                    raise ValueError("Unsupported input shape")
                
                # Convert image to numpy array with the correct dtype
                img_array = np.array(img, dtype=input_dtype)
            else:
                # Default to 512x512 if input shape is not provided
                img = img.resize((512, 512))
                # Default to float32 if input dtype is not provided
                img_array = np.array(img, dtype=np.float32)

            return img_array
        
        # Load test and reference images
        test_img = preprocess_image(test_image)
        reference_img = preprocess_image(self.reference_image_path)
        
        # Set model inputs
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(test_img, axis=0))
        interpreter.set_tensor(input_details[1]['index'], np.expand_dims(reference_img, axis=0))
        
        # Run inference
        interpreter.invoke()
        
        # Get similarity score (output)
        similarity_score = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        # Determine quality (good/bad) based on similarity threshold
        # Lower similarity score means more similar (= good) since I am using euclidean distance
        if similarity_score < threshold:
            classification = "good"
        else:
            classification = "bad"
    
        return classification, similarity_score

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> CaptureAllResult:
        if not camera_name == self.camera_name:
            raise ValueError(f"Camera name {camera_name} does not match the configured camera name {self.camera_name}")

        image = None
        if return_image:
            image = await self.camera.get_image()

        classifications = None
        if return_classifications:
            if image:
                if isinstance(image, ViamImage):
                    test_image = viam_to_pil_image(image)
                classification, similarity_score = self.test_image_quality(test_image, self.tolerance)
            classifications = [Classification(class_name=classification, confidence=similarity_score)]

        if return_detections:
            self.logger.error("`return_detections` is not implemented")
        
        if return_object_point_clouds:
            self.logger.error("`return_object_point_clouds` is not implemented")

        await self.camera.close()

        return CaptureAllResult(
            image=image,
            classifications=classifications,
            detections=None
        )

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Detection]:
        self.logger.error("`get_detections_from_camera` is not implemented")
        raise NotImplementedError

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Detection]:
        self.logger.error("`get_detections` is not implemented")
        raise NotImplementedError

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Classification]:
        self.logger.error("`get_classifications_from_camera` is not implemented")
        raise NotImplementedError

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Classification]:
        self.logger.error("`get_classifications` is not implemented")

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[PointCloudObject]:
        self.logger.error("`get_object_point_clouds` is not implemented")
        raise NotImplementedError

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> Vision.Properties:
        return Vision.Properties(
            classifications_supported=True,
            detections_supported=False,
            object_point_clouds_supported=False
        )
        
    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Mapping[str, ValueTypes]:
        self.logger.error("`do_command` is not implemented")
        return {}
