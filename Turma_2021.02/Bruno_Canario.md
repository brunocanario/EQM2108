Trabalho final da disciplina EQM2108

# Classificação de membranas polímericas utilizando modelagem por floresta aleatória

Aluno: Bruno Canario

Professora: Amanda Lemette

## Resumo
O presente trabalho propõe uma modelagem utilizando o modelo de machine learning Árvore Aleatória (Random Forest) para classificar tipos de membranas de acordo com o material utilizado na confecção das mesmas, usando como variáveis de entrada a permeabilidade e a seletividade. As membranas são específicas para o processo de permeabilidade de CO2 em gás natural, dessa forma são avaliadas a permeabilidade de CO2 e a seletividade de CO2 em relação a CH4.


## Introdução
Membranas com polímeros como camada seletiva têm sido amplamente utilizadas para a separação de misturas de gases, incluindo aqueles de relevância chave para a energia e o meio ambiente. O desenvolvimento de novos polímeros com melhor permeabilidade e seletividade ao gás aumentaria a eficiência das separações por membrana de gás de interesse industrial. Os polímeros foram desenvolvidos para vários fins, incluindo recuperação de hidrogênio durante a preparação de amônia (H2 de N2), enriquecimento de oxigênio ou nitrogênio do ar (O2 de N2); e adoçamento de gás natural ou aprimoramento do biogás (CO2 de CH4).

Métodos de aprendizado de máquina (ML) foram desenvolvidos e aplicados a polímeros para prever propriedades, incluindo temperatura de transição vítrea, constantes dielétricas, a permeabilidade de gases de polímeros e a descoberta de novos polímeros funcionais. No entanto, a permeabilidade ao gás do mesmo polímero é frequentemente medida sob diferentes condições, por exemplo, diferentes tratamentos com solvente ou grau de envelhecimento, e os modelos de ML baseados em impressões digitais de polímero não podem distinguir a diferença entre essas condições. O banco de dados de membrana de separação de gás polimérico geralmente contém dados para o mesmo polímero testado em condições diferentes, em laboratórios diferentes com instrumentos diferentes, e um modelo de ML baseado puramente na estrutura química por si só não seria suficiente para preencher os valores ausentes para permeabilidade ao gás.

Uma forma alternativa de imputar o banco de dados é prever a permeabilidade de gases desconhecidos com base em dados para gases com permeabilidade conhecida. Porém, como os coeficientes de permeabilidade do gás de logaritmo Pi e Pj dos gases i e j são fortemente correlacionados, é plausível prever a permeabilidade do gás i usando os dados de permeabilidade para outros gases sem exigir quaisquer informações sobre a estrutura molecular dos polímeros ou condições experimentais. Nesse artigo foi desenvolvido um modelo utilizando floresta aleatória (Random Forest) para classificar os diferentes tipos de membranas poliméricas utilizando somente os dados de permeabilidade e seletividade, imputando que as diferentes condições de medição não influenciam na permeabilidade dos gases.

 
## Metodologia
A base de dados utilizada foi retirada do site da Membrane Society of Australasia [1], o qual funciona como um depósito colaborativo de dados resultantes de trabalhos envolvendo membranas ao redor do mundo. Dentro do mesmo é possível selecionar o par de substâncias sobre as quais deseja-se saber os dados de permeabilidade e seletividade. Para o atual trabalho, que visa o estudo da permeabilidade de gás carbônico (CO2) em gás natural, o par escolhido foi CO2/CH4. Sendo assim, a seletividade CO2/CH4 e a permeabilidade de CO2 são usadas como variáveis de entrada para classificar os diferentes materiais que constituem as membranas.	Os dados estão representados no site também em forma de gráfico conforme a Figura 1 abaixo, sendo separadas em cores diferentes as 26 classes de materiais poliméricos que constituem as membranas. 

![image](https://user-images.githubusercontent.com/95252686/144330342-37679d89-65ae-4acd-b303-23fe2d5bd1ec.png) width = 500 heigth = 500
Figura 1 - Gráfico com todos os dados utilizados

As classes, as quais foram codificadas em valores numéricos de 0 a 25 para facilitar a identificação, possuem ao todo 1011 dados coletados. Entretanto, as mesmas possuem grandes desbalanços na quantidade de dados de cada uma, como é possível ver na Tabela 1 abaixo, onde a classe 11 possui muito mais dados que as demais. 

Tabela 1 - Quantidade de dados por classe de material das membranas
TABELAAAAAAAAAAAAAAAAAAAAAAAAA

Esse desbalanço torna a modelagem bastante complicada, comprometendo a capacidade preditiva do modelo. Portanto, após algumas tentativas iniciais de modelar dessa forma e obtenção de métricas não satisfatórias do conjunto de teste, foi tomada uma medida para contornar esse problema. A classificação passou a se basear na presença ou ausência da classe 11, ou seja, uma classificação binária onde a presença da classe 11 representa o número 1 e a ausência da classe 11 representa o número 0. Apesar de ainda prevalecer um desbalanço, a predição do modelo fica facilitada tendo somente dois grandes grupos.

Após a codificação das classes em 0 e 1, os dados foram separados em variáveis de entrada e de saída e então divididos em conjuntos de treinamento, validação e teste. Primeiro os dados foram separados em treino e teste na proporção de 90/10, resultando em 909 dados de treino e 102 de teste. Os dados anteriormente separados para treino foram separados novamente para formar o conjunto de validação. Então os 909 dados foram separados em uma proporção de 80/20, resultando em 727 dados que realmente será usado como conjunto de treino e 182 dados para a validação.

A partir desse momento, foi necessária a definição de alguns dos hiperparâmetros do modelo de Random Forest para que sejam otimizados e assim apresentar uma combinação dos mesmos que gere resultados satisfatórios para então o modelo ser treinado. Dentre os diversos hiperparâmetros existentes, foram escolhidos somente “max_depth”, “min_samples_split” e “min_samples_leaf” para serem otimizados, enquanto os demais foram mantidos nos valores padrões já definidos pelo modelo. O hiperparâmetro “max_depth” representa a profundidade máxima da árvore, “min_samples_split” representa o número mínimo de amostras necessárias para dividir um nó em dois e “min_samples_leaf” representa o número mínimo de amostras necessárias em uma folha (nó final da árvore).

A otimização foi feita usando a biblioteca “optuna” do Python, a qual faz simulações de treinamento com diferentes valores dos hiperparâmetros e avalia o desempenho do modelo comparando com os dados de validação, que servem para checar se as respostas estão corretas. Os valores testados são baseados em um intervalo escolhido para cada um dos hiperparâmetros e, baseado na melhor combinação, ou seja, o conjunto de valores que apresentou o melhor desempenho, são determinados os parâmetros ótimos. No caso o “max_depth” foi testado com valores de 1 a 8, “min_samples_split” com valores entre 2 e 4 e “min_samples_leaf” entre 2 e 5.

Em seguida, é criado o modelo de Random Forest com a melhor combinação dos hiperparâmetros, e então é treinado o modelo e a árvore é criada.

Após criado e treinado o modelo, o mesmo é avaliado de acordo com certas métricas. Primeiramente é considerada a accuracy (acurácia), precision (precisão), recall (revocação) e F1-Score tanto para a fase de treinamento quanto para a fase de teste. Para entender o que é isso, é necessário entender o conceito de True Positive, True Negative, False Positive e False Negative. Quando o modelo tenta prever um resultado positivo ou negativo e ele acerta, é chamado de True Positive e True Negative. Quando o modelo tenta prever um resultado positivo e ele erra, ou seja, predizendo como negativo, esse tipo de erro é chamado False Negative. E quando o modelo tenta prever um resultado negativo e ele erra, ou seja, predizendo como positivo, esse tipo de erro é chamado False Positive. Isso está evidenciado na Figura 2 abaixo.


![image](https://user-images.githubusercontent.com/95252686/144332782-72bb5f8b-6840-4fea-a285-cfea2304beda.png)
Figura 2 - True Positive, True Negative, False Positive e False Negative


Dessa forma a accuracy, precision, recall e f1-score são calculados conforme as equações 1 a 4 abaixo:

TABELAAAAAAAAAAAAAAAAAAAAAAA


Após o cálculo dessas métricas para o treinamento e para o teste, foi produzido matrizes de confusão para o treinamento e teste tanto de forma percentual quanto com a quantidade absoluta de dados. Em seguida, foi produzida curva de aprendizagem, onde é comparada a performance do treinamento com a do teste ao longo de todo o processo de aprendizagem. Por último foi construída a curva ROC e a curva Precision-Recall e o resultado das duas curvas será comparado.


## Resultados e Discussões
Após a separação dos dados entre conjunto de treinamento, de validação e teste, o primeiro passo antes da criação do modelo e do processo de aprendizagem através do treinamento é a otimização dos hiperparâmetros. Através do uso da biblioteca “optuna” do Python foi obtido na Figura 3 abaixo o histórico de resultados de todas as possíveis combinações. Das 100 tentativas apenas 14 atingiram o melhor desempenho, que foi de 0.802, e com isso pode-se dizer que há algumas possibilidades de combinações a serem utilizadas.


![image](https://user-images.githubusercontent.com/95252686/144334637-5497329b-120a-4d60-a635-20d7e71835dd.png)


<center><img scr= "//github.com/brunocanario/EQM2108/blob/b7309230157fa22eeac6eaf1256ddd3b0160b773/Turma_2021.02/Imagens/Historico%20de%20otimiza%C3%A7%C3%A3o%20optuna.png?raw=true" width = 600 height = 500 />
	
Figura 3 - Histórico da otimização dos hiperparâmetros


Como é possível ver na Figura 4 abaixo, o max_depth apresentou menor flexibilidade de valores do que os outros dois parâmetros, já que somente em max_depth = 7 obteve-se desempenho máximo. Em contrapartida, em min_samples_leaf e min_samples_split foi visto que o melhor desempenho foi obtido em mais de uma possibilidade. Isso evidencia uma maior importância do max_depth para o resultado final em comparação com os outros dois hiperparâmetros. Isso fica evidente na Figura 5, onde é utilizada a função optuna.visualization.plot_param_importances (), a qual mostra a importância de cada hiperparâmetro e o max_depth se confirma como o mais importante com 0,82, enquanto os demais demostram somente 0,16 e 0,01.

Figura 4 – SLICE PLOT
Figura 5 – HYPERPARAMETER IMPORTANCES

A primeira iteração é a que fica salva como ótima para o “optuna”, e nesse caso foi max_depth = 7, min_samples_leaf = 4 e min_samples_split = 4. Então foi essa a combinção utilizada no modelo criado. Após criado e treinado o modelo, a árvore criada foi a apresentada na Figura 6.

Figura 6 – Árvore construída

Agora para avaliar o modelo foi primeiramente calculado a accuracy, precisiom, recall e F1-Score do treinamento e do teste. Para o treinamento, a accuracy foi de 0.8789546 enquanto que para o teste foi de 0.75490196. Já para a precision, recall e F1-Score os resultados estão apresentados nas tabelas 2 e 3 abaixo respectivamente para treinamento e teste.

Tabela 2 - Precision, Recall e F1-Score do treinamento
TABELAAAAAAAAAAAAAAA

Tabela 3 - Precision, Recall e F1-Score do teste
TABELAAAAAAAAAAAAAAA 


Nota-se que os resultados não são excelentes, porém podem ser considerados satisfatórios. Cabe destacar que houve uma diferença significativa entre os recalls das duas classes, o que implica que o número de False Negatives foi bastante discrepante, havendo muito mais para a classe 1 do que para a classe 0. Ou seja, quando era para o modelo classificar como 1 ele classificou como 0 em maior frequência do que o inverso. Além disso, se for considerar a média com pesos para o recall ela foi maior que a média sem pesos. Isso também aconteceu para as médias com peso do F1-Score.


As matrizes de confusão estão descritas abaixo nas Figuras 7, 8, 9 e 10 e é possível verificar que novamente o modelo acertou a predição da classe 0 muito mais precisamente do que da classe 1. No treinamento foi melhor do que no teste, no qual o modelo teve mais erros do que acertos, inclusive.
	
Figura 7 – Matriz de confusão com valores absolutos do treinamento 
Figura 8 – Matriz de confusão com valores absolutos do teste
Figura 9 – Matriz de confusão com valores percentuais do treinamento 
Figura 10 – Matriz de confusão com valores percentuais do teste


A curva de aprendizagem, representada na Figura 11, apresentou o comportamento esperado, onde o desempenho do treinamento é mais alto no começo da aprendizagem e vai diminuindo com o tempo, e o desempenho do teste começa mais baixo e tende a ir aumentando com o tempo, e assim a as curvas se aproximam.

Figura 11 – Curva de Aprendizagem


Por fim, a curva ROC e a curva Precision-Recall estão representadas na Figuras 12 e 13 abaixo. 

Figura 12 – Curva ROC
Figura 13 – Curva Precision-Recall


Ambas as curvas possuem suas particularidades e suas características específicas. A Curva ROC é um gráfico que apresenta no eixo X a taxa de False Positives (equação 5) e no eixo Y a taxa de True Positives (equação 6). O objetivo do mesmo é descobrir a habilidade que o modelo possui de conseguir prever os resultados. Essa habilidade é quantificada através da área abaixo da curva. Um modelo que não possui habilidade alguma e, portanto, que não consegue distinguir entre as classes, possui uma área abaixo da curva de 0,5. Ou seja, é construída uma reta diagonal que vai da extremidade inferior esquerda até a extremidade superior direita. Então, a curva ROC que apresenta uma curvatura acima da diagonal já apresenta certa habilidade.

TABELAAAAAAAAAA


Já em relação à curva precision-recall, ela plota o recall no eixo X e a precision no eixo Y. Porém, diferentemente da curva ROC, a curva precision-recall é mais útil para situações onde a classificação binária possui um desbalanceamento na quantidade de dados entre as duas classes. Em especial há muitos exemplos onde o número de dados da classe 0 é muito maior do que na classe 1 e nesse caso interessa-se menos na habilidade do modelo de prever a classe 0 (True Negatives). Isso porque no cálculo de precision e recall não se faz uso de True Negatives, estando apenas interessado na predição correta na classe minoritária (classe 1). Os True Negatives estão presentes na curva ROC dentro da taxa de False Positives, o que é evitado na curva precision-recall e por isso se torna ideal para esse tipo de situação.

	Normalmente a curva ROC apresenta uma perspectiva mais otimista sobre a habilidade do modelo de previsão quando os dados estão desbalanceados, e isso pode levar a interpretações incorretas e conclusões precipitadas sobre a performance do modelo. Isso é possível ser visto nos gráficos acima e nas métricas que quantificam a área abaixo da curva e, portanto, a habilidade do modelo. Para a curva ROC, a área abaixo da curva possui valor de 0,899, já para a curva precision-recall o valor foi de 0,797. Outra métrica de quantificação de qualidade para a curva precision-recall é o valor de f1, que é a média harmônica da precision e do recall, e apresenta valores entre 0 e 1. No caso do modelo criado ela teve um valor de 0,510, o que é bastante baixo. Isso comprova como a curva ROC mascara de certa forma o desempenho do modelo, que acaba sendo mais realista através curva precision-recall.


## Conclusões
Com isso é possível concluir que o modelo ainda precisa de melhorias para conseguir apresentar resultados melhores e mais precisos. Principalmente na avaliação da fase de teste é possível uma queda mais acentuada no desempenho do modelo, e uma dificuldade maior de prever os resultados corretos. Em futuros trabalhos vale a pena tentar investigar outros hiperparâmetros e a importância dos mesmos, além de ser válida a tentativa de agrupar as classes de materiais de polímeros de outra forma, em grupos que possuem características parecidas. Alternativamente, talvez seja interessante a busca pela criação de dados artificialmente para conseguir balancear a base de dados entre as classes.


## Referências
