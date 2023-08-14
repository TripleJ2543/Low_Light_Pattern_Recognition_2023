# Low_Light_Pattern_Recognition

"Low-Light Image Enhancement Using Gamma Correction Prior in Mixed Color Spaces," in revision   
[Jong Ju Jeon](https://github.com/TripleJ2543)(triplej@pusan.ac.kr), Il Kyu Eom*(ikeom@pusan.ac.kr)(https://sites.google.com/view/ispl-pnu)   

[Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4362440)


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
Jong Ju Jeon, Il Kyu Eom,   
Low-Light Image Enhancement Using Gamma Correction Prior in Mixed Color Spaces,   
in revision,   
2023,   
https://doi.org/.   
(https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4362440)
Keywords: Low-light image enhancement; Inverted image; Atmospheric light; Normalization; Transmission map; Retinex model; HSV color space; HSI color space; Gamma correction   
