It backend are compose for many scrips


main.py
models.py
depth_utils.py
config.py


The logic

First in main it recive a image in png or jpg, then once we have the picture we convert it into a matrix with open CV
(RGB), then we call the functions of our models (YOLO OBB for boundiboxes detection), (YOLO segmentation for rejilla segemntation)
and Depth Anything model to know the depth of the products. Use cm insted of m is so important for the model
we need cm beacuse we will create an math operations with this cm 

How auxiliar functions works
------------------------------------------------------------------------------------------------------------------------------
config.py
 This scrip containts important values such as how many product are in total if it is full, also we have the measure of each product
 for example how many space ocupied a clasic magnum.
 It contains the path of YOLO models, matriz of weight of neuronal networks

best.pt -> is matriz of weight for the modelo YOLO OBB
segmentation.pt -> is a matrix of weights for modelo YOLO segmentation 
Note: As depth anything use dinov2 under the wood it dont need file of weight for neuronal networks


models.py
    It call the functions for use models, read path from config.py and call the models with python, if GPU is available
    it will use overwiese use CPU, 

depth_utils.py
    I have many functions such as: 
        - calcular_mapa_profundidad_cm()
        - extraer_profundidad_bbox()
        -  extraer_coordenadas_rejillas_knn_pura()


    calcular_mapa_profundidad_cm()
        This functions will give me a map of depth, it recive a img and give me a matrix with depth values depend of the position 
        It will convert into cm.

    extraer_profundidad_bbox()
        This will find the mean depath of the YOLO OBB detections, with depth matrix we create a mean of the product,
        