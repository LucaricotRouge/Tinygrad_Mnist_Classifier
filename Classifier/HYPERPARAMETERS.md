## Results and discussions about the learning curve and the hyperparameters chosen : 


### MLP Model : 

![mlp0](../Results/mlp/mlp0.png)
lr: 2.00e-02  loss: 0.30  accuracy: 93.03%
Test primaire pour visualiser la courbe d'apprentissage avec des hyperparamètres standards dont un STEPS de 100.

![mlp1](../Results/mlp/mlp1.png)
lr: 2.00e-02  loss: 0.37  accuracy: 92.14

Test avec un BATCH plus petit pour vérifier si le modèle apprendrait mieux mais dans la mesure où il n'overfit pas ce n'est pas si nécessaire. 

![mlp2](../Results/mlp/mlp2.png)
lr: 2.00e-02  loss: 0.22  accuracy: 95.78%
L'augmentation du STEPS malgré un BATCH plus faible a eu un impact important sur l'apprentissage du modèle. Pour un BATCH réduit, augmenter le nombre d'itération devrait laisser au modèle suffisamment d'échantillons d'entraînementa afin que le modèle puisse apprendre. 

![mlp3](../Results/mlp/mlp3.png)
lr: 1.00e-03  loss: 0.76  accuracy: 88.15%:

 En testant le modèle sur l'interface Web, on peut constater des faiblesse sur la manière de gérer les dessins non centré et a taille réduite par exemple. C'est pourquoi on va donner plus de données aléatoires à notre modèle en augmentant les hyperparamètres d'ANGLE, de SCALE et de SHIFT. Néanmoins on observe que l'accuracy baisse et la loss augmente suite à ces changements. On peut supposer que le paramètre d'ANGLE a une influence importante sur la compréhension du modèle sur les chiffres car un angle trop aléatoire ou différent ne permettrait pas d'apprendre efficacement les caractéristiques des chiffres ce qui pourrait causé des confusions entre un 1 et un 7 par exemple (qui selon l'angle peut sembler similaire ?). Il est également possible que le modèle n'a pas assez d'itération pour apprendre étant donné qu'on a augmenté drastiquement la base de données en touchant aux hyperparamètres d'ANGLE, de SCALE et de SHIFT. On peut ainsi penser à augmenter le BATCH mais aussi à changer moins drastiquement l'ANGLE, la SCALE et le SHIFT et c'est ce nous allons tenter de faire au prochain entraînement.  

![mlp4](../Results/mlp/mlp4.png)
lr: 2.00e-02  loss: 0.22  accuracy: 95.75%

Nous obtenons des résultats plus corrects. 

![mlp5](../Results/mlp/mlp5.png)
lr: 2.00e-02  loss: 0.38  accuracy: 94.84%

Augmenter le SHIFT a baisser l'accuracy et augmenté la loss mais de manière pas si drastique par rapport à l'augmentation forte de la loss. Néanmoins la courbe présente une croissance avec un "second" pique d'apprentissage par une difficulté d'apprendre, dû à un BATCH trop petit par rapport à la nouvelle base de données plus grande dû au SHIFT mis de 0.1 à 3. Nous allons augmenter le nombre de BATCH pour ces raisons et observer les changements. 


![mlp6](../Results/mlp/mlp6.png)
lr: 2.00e-02  loss: 0.26  accuracy: 95.88%

Le résultat est à nouveau plus correct 

![mlp7](../Results/mlp/mlp7.png)
lr: 2.50e-02  loss: 0.27  accuracy: 96.07%



### CNN Model : 

![cnn0](../Results/cnn/cnn0.png)
lr: 1.00e-03  loss: 1.07  accuracy: 88.38%

Test primaire pour visualiser la courbe d'apprentissage avec des hyperparamètres standards dont un STEP de 100.

![cnn1](../Results/cnn/cnn1.png)
lr: 1.00e-03  loss: 1.04  accuracy: 88.38%

La diminution de la taille du BATCH ne semble pas impacter la loss et l'accuracy. Néanmoins sur un plus grand nombre de STEPS on peut supposer que le modèle continuera d'apprendre et ce de manière plus optimisée car un nombre de BATCH plus faible devrait limiter un overfitting avec un nombre de STEPS plus grand. 

![cnn2](../Results/cnn/cnn2.png)
lr: 1.00e-03  loss: 1.03  accuracy: 89.41%

La diminution du LR a permis un pique d'apprentissage plus important par un ajustement des poids moins brutals mais la loss et l'accuracy ne change quasiment pas. 

![cnn5](../Results/cnn/cnn5.png)
1.00e-03  loss: 0.34  accuracy: 95.93%

![cnn6](../Results/cnn/cnn6.png)
lr: 1.00e-03  loss: 0.45  accuracy: 93.31%

![cnn7](../Results/cnn/cnn7.png)
lr: 1.00e-03  loss: 0.51  accuracy: 93.34%

![cnn8](../Results/cnn/cnn8.png)
lr: 1.27e-02  loss: 0.07  accuracy: 98.95%