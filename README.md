# Low_Light_Pattern_Recognition

"[Low-Light Image Enhancement Using Gamma Correction Prior in Mixed Color Spaces](https://doi.org/10.1016/j.patcog.2023.110001)," Pattern Recognition
[Jong Ju Jeon](https://github.com/TripleJ2543)(triplej@pusan.ac.kr), Jun Young Park(jyp9917140@pusan.ac.kr), Il Kyu Eom*(ikeom@pusan.ac.kr)(https://sites.google.com/view/ispl-pnu)   

[Site](https://sites.google.com/view/ispl-pnu), [Paper](https://doi.org/10.1016/j.patcog.2023.110001)


### Requirements ###
1. Linux
2. Python (3.11.4)
3. scikit-image (0.20.0)
4. opencv (4.6.0)


### Usage ###
you can just run through
```shell
python Run_LowLight_Image_Enhancement_Pixel_Adaptive.py 
    --input_dir=/path/to/your/dataset/dir/ \
    --output_dir=/path/to/save/results/ \
    --gamma_max=6                               # defaults gamma_max=6

#python Run_LowLight.py --input_dir=/path/to/your/dataset/dir/ --output_dir=/path/to/save/results/ --gamma_max=6

```

### Citation ###
Jong Ju Jeon, Jun Young Park, Il Kyu Eom,   
Low-Light Image Enhancement Using Gamma Correction Prior in Mixed Color Spaces,   
Pattern Recognition,   
2023,   
https://doi.org/10.1016/j.patcog.2023.110001.
(https://www.sciencedirect.com/science/article/pii/S0031320323006994)

Keywords: Low-light image enhancement; Inverted image; Atmospheric light; Normalization; Transmission map; Retinex model; HSV color space; HSI color space; Gamma correction   
