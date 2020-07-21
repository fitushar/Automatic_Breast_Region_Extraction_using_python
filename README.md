# Automatic_Breast_Region_Extraction_using_python
Brest region extraction from the memogram often the initaial stage to do image analysis or dignosis. This Repo contains Python Implementation of an Automatic approach to Extraction Breast Region from Memogram. All you need to do is to set the threshold based on your memogram vendor, Different vendors could have different pixel values.

Here is public dataset to try the code: https://www.dclunie.com/pixelmedimagearchive/upmcdigitalmammotomocollection/index.html


### Brest Region Extraction Algorithm works as following: Jupyter notebook have all the codes
```ruby  
   i) Load Dicom and Threshold based on your vendor
  ii) Connected Component Analaysis and Keep the Largest Mask
 iii) Erode to remove unwanted object in memo.
  iv) Label Erode Image and Replace all Non-ROI with background pixel
   v) Finally dilat the mask to recover the eroded Breast Region
```  

![Simple Preocess](https://github.com/fitushar/Automatic_Breast_Region_Extraction_using_python/blob/master/Figure/Process.png)

