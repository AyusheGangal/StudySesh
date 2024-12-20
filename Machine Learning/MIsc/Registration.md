Image registration is the process of aligning two or more images of the same scene or object so that they have the same spatial reference frame. Image registration is a fundamental task in many computer vision and image processing applications, such as medical imaging, remote sensing, and surveillance.

The goal of image registration is to find a geometric transformation that maps one image onto another. The transformation can be rigid (translation, rotation, and scaling) or non-rigid (deformation, warping). The transformation can be found using various techniques, such as feature-based methods, intensity-based methods, and hybrid methods.

Feature-based methods detect and match salient features, such as corners, edges, or blobs, in both images. Once the corresponding features are found, a geometric transformation that maps one image onto another can be estimated using methods such as least squares or RANSAC.

Intensity-based methods use image similarity metrics, such as correlation, mutual information, or normalized gradient fields, to estimate the transformation that maximizes the similarity between the two images.

Hybrid methods combine both feature-based and intensity-based approaches to achieve more accurate and robust registration.

Once the transformation is found, it can be applied to one or more images to align them with a reference image or a common coordinate system. Image registration has many practical applications, such as image fusion, image segmentation, object tracking, and image-guided interventions.

### Multi-modal image registration
In the context of image registration, "multi-modal" refers to the use of multiple imaging modalities or imaging techniques to acquire images of the same object or scene. Different imaging modalities can capture different aspects of the object or scene, such as anatomical structure, functional activity, or molecular composition, and provide complementary information. For example, magnetic resonance imaging (MRI) can provide high-resolution anatomical information, while positron emission tomography (PET) can provide functional information about metabolic activity or blood flow. By registering multiple modalities, one can integrate the information from different modalities and obtain a more comprehensive and accurate representation of the object or scene. However, multi-modal image registration is challenging, as the images may have different resolution, orientation, and intensity characteristics, which must be taken into account during the registration process.

Multi-modal image registration is the process of aligning two or more images of the same object or scene, where the images have different modalities or imaging properties, such as magnetic resonance imaging (MRI), computed tomography (CT), positron emission tomography (PET), or ultrasound. Multi-modal image registration is a challenging task, as the images may have different image resolutions, orientations, intensities, and noise levels.

The goal of multi-modal image registration is to find a transformation that maps one image onto another, while preserving the anatomical or functional information in the images. The transformation can be rigid or non-rigid, and can be found using various techniques, such as mutual information, correlation coefficient, normalized gradient fields, or feature-based methods.

Mutual information is a popular similarity metric used in multi-modal image registration, as it is insensitive to differences in image intensity and modality. Mutual information measures the statistical dependence between the pixel intensities in the two images, and maximizes the joint entropy of the images. Mutual information is not strictly considered a feature-based method for image registration, but it is often used as a similarity measure in both feature-based and intensity-based registration methods. Mutual information is a statistical measure that quantifies the degree of dependence between two variables, in this case, the intensity values of the images being registered. Mutual information can be used as a similarity measure because it is insensitive to differences in image contrast and intensity, making it well-suited for multi-modal image registration. In practice, mutual information is often used in conjunction with feature-based methods to achieve better registration accuracy.

Correlation coefficient is another similarity metric used in multi-modal image registration, which measures the linear correlation between the intensities of the two images.

Normalized gradient fields is a gradient-based similarity metric, which measures the spatial gradients of the images and their orientations.

Feature-based methods use salient features, such as corners, edges, or blobs, to estimate the transformation between the images. These features can be detected and matched in both images, and a geometric transformation that maps one image onto another can be estimated using methods such as least squares or RANSAC.

Once the transformation is found, it can be applied to one or more images to align them with a reference image or a common coordinate system. Multi-modal image registration has many applications in medical imaging, such as image fusion, radiation therapy planning, and diagnosis of diseases.

Feature-based, intensity-based, and hybrid methods are all approaches to image registration, but they differ in their underlying principles and how they achieve alignment between images. Here are some brief explanations of each method:

1.  Feature-based methods: These methods identify and match salient features between two or more images, such as edges, corners, or other distinct patterns. Feature-based methods are useful when the images being registered have little or no overlap, or when there are significant geometric differences between the images. Feature-based methods typically involve three main steps: feature extraction, feature matching, and transformation estimation.
    
2.  Intensity-based methods: These methods rely on similarity measures based on pixel or voxel intensities to find the best alignment between two or more images. Intensity-based methods work well when the images being registered have significant overlap and similar structures, and when there are minimal geometric differences between the images. Common similarity measures used in intensity-based methods include mutual information, correlation coefficient, and sum of squared differences.
    
3.  Hybrid methods: These methods combine both feature-based and intensity-based techniques to achieve better registration accuracy, especially in challenging scenarios where one method alone may not be sufficient. For example, a hybrid method might first use feature-based methods to estimate an initial transformation between images, and then refine that transformation using intensity-based methods. Hybrid methods can be more robust and accurate than either feature-based or intensity-based methods alone.
    

In summary, feature-based methods are good for registering images with significant geometric differences, intensity-based methods work well for images with significant overlap and similar structures, and hybrid methods combine the strengths of both approaches to achieve better registration accuracy in challenging scenarios.

<mark style="background: #D2B3FFA6;">What I did:</mark>
1.  User can load MSI data from an analyte .txt file (i.e. using msipy.file_service.load_analyte_data) 
2.  User can load an opitcal image from a number of file types including: .tif, .png, .jpg, etc.
3.  User can visualize the optical image and any m/z image from the analyte file in separate windows
4.  Registration
	* Resize optical image to dimensions of MSI image (found using MSIData.to_grid). This will be the "moving" image
	* (maybe sum over RGB channels to get a single 2D array for calculating the transformation matrix)
5. Get the TIC image from the MSI data (should be in MSIData.tic, otherwise just sum all spectra values at each pixel). This will be the "target" or static image.
6. Calculate the linear, "affine" transformation matrix between the downscaled optical image and the static TIC image  
7. This allows for translation, scaling, rotation, shearing, etc.
8. If possible, it should optimize to maximize "mutual information" or a related metric, not "cross-correlation", since the images will not have the same contrast mechanism
9. If there are any parameters to tune (such as metric above, or freezing to only translation, only rotation, etc.), pick sensible defaults, but expose them to the user
10. The output of the affine transformation should be a matrix (3x3 or 4x4, possibly, depending on how you do it). This matrix should be applied to the ORIGINAL FULL-SIZE optical image
11. Now, upsample the full MSI data to the dimensions of the TRANSFORMED optical image
12. Show both images on the same axes, with a st.slider to determine opacity of the MSI image. The user should be able to select ANY m/z image from the MSI dataset to overlay on the optical image, along with colormap and max/min range, etc.
13. User should be able to export the transformed optical image, as well as the overlay image figure