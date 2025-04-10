# Module siamese-network 

This module provides a Siamese Network vision service for comparing images against a reference image. It uses a Siamese Neural Network implemented with TensorFlow to determine if a test image is similar to a reference image based on a similarity threshold.

## Model brad-grigsby:siamese-network:siamese-network-vision

This vision service model implements a Siamese Network for image comparison. It loads a TensorFlow Lite model to compare images captured from a camera against a reference image, and classifies them as "good" or "bad" based on their similarity score. The model is particularly useful for quality control applications where images need to be compared against a known good reference.

### Configuration
The following attribute template can be used to configure this model:

```json
{
  "model_path": <string>,
  "reference_image_path": <string>,
  "camera_name": <string>,
  "tolerance": <float>
}
```

#### Attributes

The following attributes are available for this model:

| Name                  | Type   | Inclusion | Description                                             |
|-----------------------|--------|-----------|--------------------------------------------------------|
| `model_path`          | string | Required  | Path to the TFLite Siamese Network model               |
| `reference_image_path`| string | Required  | Path to the reference image for comparison             |
| `camera_name`         | string | Required  | Name of the camera component to capture images from    |
| `tolerance`           | float  | Optional  | Similarity threshold (default: 0.5, lower is more similar) |

#### Example Configuration

```json
{
  "model_path": "/path/to/siamese_model.tflite",
  "reference_image_path": "/path/to/reference_image.jpg",
  "camera_name": "my_camera",
  "tolerance": 0.4
}
```

### DoCommand

This model does not implement DoCommand functionality.
