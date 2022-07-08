def register(path):
        t1_path = path
        
        import os

        import elastix
        import imageio

        import elastix
        import numpy as np
        import imageio
        import os
        import SimpleITK as sitk
        def change_parameter(input_path, old_text, new_text, output_path):
            """
            replaces the old_text to the next_text in parameter files

            Parameters
            ----------
            input_path : str
                parameter file path to be changed.
            old_text : str
                old text.
            new_text : str
                new text.
            output_path : str
                changed paramter file path.
            Returns
            -------
            None.
            """
            #check if input_path exists
            if not os.path.exists(input_path):
                print(input_path + ' does not exist.')

            a_file = open(input_path)
            list_of_lines = a_file.readlines()
            for line in range(0,len(list_of_lines)):
                if (list_of_lines[line] == old_text):
                    list_of_lines[line] = new_text

            a_file = open(output_path, 'w')
            a_file.writelines(list_of_lines)
            a_file.close()



        # IMPORTANT: these paths may differ on your system, depending on where
        # Elastix has been installed. Please set accordingly.

        #ELASTIX_PATH = os.path.join('elastix-5.0.1-linux/bin/elastix')
        #TRANSFORMIX_PATH = os.path.join('elastix-5.0.1-linux/bin/transformix')
        ELASTIX_PATH = os.path.join('/user/hugok/Projects/VALDO/elastix/elastix-5.0.1-linux/bin/elastix')
        TRANSFORMIX_PATH = os.path.join('/user/hugok/Projects/VALDO/elastix/elastix-5.0.1-linux/bin/transformix')

        if not os.path.exists(ELASTIX_PATH):
            raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
        if not os.path.exists(TRANSFORMIX_PATH):
            raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')

        # Make a results directory if non exists
        if os.path.exists('results') is False:
            os.mkdir('results')

        # Define the paths to the two images you want to register
        target_dir = os.path.join(t1_path)  #### ---------------------------> path[x] before 
        moving_dir = os.path.join( 'example_data', 'mni.nii')
        moving_mask_dir = os.path.join('example_data', 'Prevalence_map-csv.nii.gz')
        output_dir='results'


        # Define a new elastix object 'el' with the correct path to elastix
        el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

        # Register the moving image to the target image with el →
        el.register(
            fixed_image=target_dir,
            moving_image=moving_dir,
            parameters=[os.path.join( 'example_data', 'affine.txt'), os.path.join('example_data', 'bspline.txt')],
            output_dir=os.path.join('results'))
        # NOTE: two TransformParameters files will come out of this. Check which one to use for transformix. One file calls the other, so only provide one.

        # Find the results
        transform_path = os.path.join(output_dir, 'TransformParameters.1.txt')
        result_path = os.path.join(output_dir, 'result.1.nii')


        param_path=transform_path
        for i in range(len(param_path)):
            old_text = '(FinalBSplineInterpolationOrder 3)\n'
            new_text = '(FinalBSplineInterpolationOrder 0)\n'
            change_parameter(param_path , old_text, new_text, param_path)


        # Feed the directory of the parameters from the registration to a tr → 
        tr = elastix.TransformixInterface(parameters=transform_path,
                                        transformix_path=TRANSFORMIX_PATH)                             
        tr.transform_image(moving_mask_dir, output_dir=r'results')


        # Apply it to the moving prostate segmentation → 
        transformed_image_path = tr.transform_image(moving_mask_dir, output_dir=r'results')

        moving_img_mask = sitk.GetArrayFromImage(sitk.ReadImage(transformed_image_path))
        #print(moving_img_mask)


        img1= sitk.ReadImage('results/result.nii')

        Im = img1
        BinThreshImFilt = sitk.BinaryThresholdImageFilter()
        BinThreshImFilt.SetLowerThreshold(1)
        BinThreshImFilt.SetOutsideValue(0)
        BinThreshImFilt.SetInsideValue(1)
        BinIm = BinThreshImFilt.Execute(Im)

        sitk.WriteImage(BinIm, 'results/prevalence_map.nii.gz')