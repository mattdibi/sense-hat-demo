import numpy as np
import json

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "INPUT0")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:

            acc_x      = pb_utils.get_input_tensor_by_name(request, "ACC_X").as_numpy()
            acc_y      = pb_utils.get_input_tensor_by_name(request, "ACC_Y").as_numpy()
            acc_z      = pb_utils.get_input_tensor_by_name(request, "ACC_Z").as_numpy()
            gyro_x     = pb_utils.get_input_tensor_by_name(request, "GYRO_X").as_numpy()
            gyro_y     = pb_utils.get_input_tensor_by_name(request, "GYRO_Y").as_numpy()
            gyro_z     = pb_utils.get_input_tensor_by_name(request, "GYRO_Z").as_numpy()
            humidity   = pb_utils.get_input_tensor_by_name(request, "HUMIDITY").as_numpy()
            magnet_x   = pb_utils.get_input_tensor_by_name(request, "MAGNET_X").as_numpy()
            magnet_y   = pb_utils.get_input_tensor_by_name(request, "MAGNET_Y").as_numpy()
            magnet_z   = pb_utils.get_input_tensor_by_name(request, "MAGNET_Z").as_numpy()
            pressure   = pb_utils.get_input_tensor_by_name(request, "PRESSURE").as_numpy()
            temp_hum   = pb_utils.get_input_tensor_by_name(request, "TEMP_HUM").as_numpy()
            temp_press = pb_utils.get_input_tensor_by_name(request, "TEMP_PRESS").as_numpy()

            out_0 = np.array([magnet_x, magnet_z, magnet_y, acc_y, acc_x, acc_z, pressure, temp_press, temp_hum, humidity, gyro_x, gyro_y, gyro_z]).transpose()

            #                MAGNET_X   MAGNET_Z   MAGNET_Y     ACC_Y     ACC_X     ACC_Z    PRESSURE   TEMP_PRESS   TEMP_HUM   HUMIDITY    GYRO_X    GYRO_Y    GYRO_Z
            min = np.array([-1.8199598e+01, 1.3105305e+01, -3.1945429e+01, -1.0663300e-01, -3.1721205e-01, 8.5927510e-01, 9.9792896e+02, 3.4666668e+01, 3.6153427e+01, 2.0175404e+01, -2.1469840e+00, -1.3511374e+00, -5.0122184e-01])
            max = np.array([ 3.7974422e+00, 1.7207603e+01, -2.4966690e+01, 1.1758627e-01, 2.4764843e-01, 1.1285601e+00, 9.9812500e+02, 3.6097916e+01, 3.7271065e+01, 2.3081772e+01, 1.8424674e+00, 7.2500044e-01, 4.4410592e-01])

            # MinMax scaling
            out_0_scaled = (out_0 - min)/(max - min)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor("INPUT0",
                                           out_0_scaled.astype(output0_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
