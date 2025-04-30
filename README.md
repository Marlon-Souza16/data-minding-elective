# data-minding-elective

## Alunos: 
- Lucas Willian de Souza Serpa
- Marlon de Souza

  ---

1. Em análise de anomalia/outliers com mais de uma variável pode-se aplicar a técnica de agrupamentos
(clustering), dado o dataset sfo_2018_data file_final_Weightedv2 (publicado no Teams), responda as questões:
(2,0)
a. Existe um grupo incomum de passageiros que não se enquadra no perfil típico de cliente do
aeroporto?
i. Qual é o tamanho do cluster em percentagem aos passageiros do aeroporto e qual é o perfil
do grupo?
Para tanto, são sugeridas as seguintes variáveis: score de satisfação NETPRO, variáveis demográficas:
Q20Age (idade), Q21Gender (gênero), Q22Income (renda). Variáveis comportamentais: Q23FLY (frequência
de voos), Q5TIMESFLOW (experiência de voos) e Q6LONGUSE (há quanto tempo voam pelo aeroporto de
SFO) .
- Resposta: Baseado no codigo gerado em python para analise do dataset, podemos dizer que:
  a. Existe um grupo incomum de passageiros que não se enquadra no perfil típico de cliente do aeroporto?
Sim, existe um grupo incomum de passageiros, identificado como o cluster 2. Esse cluster representa uma pequena porção dos passageiros do aeroporto, com aproximadamente 5,7% dos passageiros, de acordo com a distribuição dos clusters.

i. Qual é o tamanho do cluster em percentagem aos passageiros do aeroporto e qual é o perfil do grupo?
Tamanho do Cluster em Percentagem:

O cluster incomum, identificado como o Cluster 2, representa 5,7% do total de passageiros.

Perfil do Cluster Incomum:

O perfil do grupo (Cluster 2) é o seguinte:

NETPRO (score de satisfação): A média de satisfação desse grupo é 9,94, com uma variação que vai de 0 a 11. Isso sugere que, em média, o grupo possui uma satisfação relativamente alta, embora existam alguns passageiros com satisfação muito baixa.

Idade (Q20Age): A idade média do grupo é 0,58, o que pode representar uma faixa etária jovem, já que 0 pode significar uma idade mais baixa (a codificação pode representar faixas etárias específicas).

Gênero (Q21Gender): A média de 0,21 indica que a maior parte dos membros deste grupo são do gênero masculino, dado que a codificação dos gêneros é comumente feita assim (masculino = 0, feminino = 1).

Renda (Q22Income): A média de 0,075 sugere que a maioria dos passageiros deste cluster tem uma faixa de renda mais baixa, com a codificação indicando diferentes faixas de renda.

Frequência de voos (Q23FLY): A média é 0,09, o que indica que a maioria dos passageiros desse grupo voa poucas vezes, possivelmente mais esporadicamente.

Experiência de voos (Q5TIMESFLOWN): A média é 2,39, sugerindo que esses passageiros têm um número relativamente baixo de voos em sua experiência (em média, voam cerca de 2-3 vezes).

Tempo de uso do aeroporto (Q6LONGUSE): A média é 2,5, indicando que os membros deste grupo têm uma experiência moderada com o aeroporto de SFO, com uso variando entre 1 e 4 anos.

Conclusão:
O Cluster 2 representa um grupo incomum de passageiros, com características de baixa frequência de voos, experiência moderada com o aeroporto e renda mais baixa. Apesar de sua alta satisfação média (NETPRO), esse grupo é significativamente menor que os outros clusters (representando apenas 5,7% do total de passageiros), o que o torna incomum em relação aos outros grupos.

Esse grupo pode ser interessante para futuras campanhas de marketing ou para a análise de estratégias de melhorias específicas no aeroporto.

2. Regras de Associação são utilizadas para buscar elementos que consequentemente implicam na presença de
outros elementos em uma transação. Tais regras são utilizadas em diferentes áreas como marketing, vendas,
sistema de recomendação, prevenção de crimes, etc. Portanto, gere regras de associação para um problema
diferente de transações de itens de compras e apresente (2,0):
a. Problema;
b. Dados utilizado para modelagem do problema (dataset);
c. Passos utilizados para geração de regras;
d. Regras Geradas.
Pode-se experimentar os algoritmos Eclat ou FP-Growth como alternativa ao Apriori.
- Resposta: Com base no codigo gerado, podemos dizer que:

- a. Problema
O problema em questão é analisar os fatores relacionados à satisfação de passageiros em um aeroporto e como essas variáveis influenciam umas às outras. A ideia é usar regras de associação para identificar padrões entre as características dos passageiros (como idade, gênero, frequência de voos) e suas respostas a diferentes questões, como satisfação com o atendimento.

- b. Dados utilizados para modelagem do problema (dataset)
O dataset utilizado é um arquivo com informações de passageiros do aeroporto, incluindo dados como:

NETPRO: Índice de recomendação de um serviço (NPS - Net Promoter Score).

Q20Age: Idade do passageiro.

Q21Gender: Gênero do passageiro.

Q22Income: Faixa de renda do passageiro.

Q23FLY: Frequência de voos.

Q5TIMESFLOWN: Quantidade de voos realizados.

Q6LONGUSE: Hábitos de uso do aeroporto.

- c. Passos utilizados para geração de regras
Os passos que foram seguidos para gerar as regras de associação foram:

Carregamento do Dataset: Os dados foram carregados a partir de um arquivo CSV com o uso da função pd.read_csv.

Seleção de Variáveis: Foram selecionadas algumas variáveis de interesse relacionadas a satisfação e características do passageiro (idade, gênero, renda, frequência de voos, entre outras).

Tratamento de Dados Faltantes: As linhas com dados faltantes foram removidas.

Codificação de Variáveis Categóricas: Variáveis como gênero, renda e frequência de voos foram transformadas em valores numéricos (por meio de codificação LabelEncoder).

Binarização dos Dados: Para aplicar o algoritmo de regras de associação, as variáveis foram transformadas para valores binários (0 ou 1), de acordo com thresholds definidos para diferentes variáveis.

Geração de Itemsets Frequentes: Foi utilizado o algoritmo FP-Growth para encontrar os itemsets frequentes no conjunto de dados binarizado.

Geração de Regras de Associação: Com os itemsets frequentes, foi gerado o conjunto de regras de associação usando a métrica de lift.

- d. Regras Geradas
Aqui estão algumas das regras geradas pelo algoritmo FP-Growth:

Regra 1:

Antecedente: Quando o passageiro tem uma determinada idade (Q20Age).

Consequente: O passageiro tende a voar frequentemente (Q23FLY).

Suporte: 91.53% dos dados têm essa associação.

Confiança: 97.42% de certeza de que a presença de uma variável implica na outra.

Lift: 1.05, indicando uma leve associação entre idade e a frequência de voos.

Regra 2:

Antecedente: Quando o passageiro voa frequentemente (Q23FLY).

Consequente: O passageiro tem uma determinada idade (Q20Age).

Suporte: 91.53% dos dados têm essa associação.

Confiança: 98.39% de certeza de que a presença de uma variável implica na outra.

Lift: 1.05.

Regra 3:

Antecedente: Quando o passageiro tem uma determinada renda (Q22Income).

Consequente: O passageiro tende a voar frequentemente (Q23FLY).

Suporte: 76.57% dos dados têm essa associação.

Confiança: 82.32% de certeza de que a presença de uma variável implica na outra.

Lift: 1.07, mostrando uma leve associação entre renda e frequência de voos.

Regra 4:

Antecedente: Quando o passageiro tem um determinado gênero (Q21Gender).

Consequente: O passageiro tem uma determinada renda (Q22Income).

Suporte: 76.21% dos dados têm essa associação.

Confiança: 81.13% de certeza de que a presença de uma variável implica na outra.

Lift: 1.05, indicando uma leve associação entre gênero e renda.

Essas são algumas das regras geradas, com base nas variáveis selecionadas. Essas regras indicam padrões interessantes que podem ser úteis para entender as características e comportamentos dos passageiros em relação à sua satisfação com o aeroporto.

Conclusão:
Sim, com esses resultados, é totalmente possível responder à questão. Você usou FP-Growth para gerar as regras de associação, que relacionam diferentes características dos passageiros com a frequência de voos e outras variáveis. Isso poderia ser aplicado em áreas como marketing, vendas, ou sistemas de recomendação no contexto de aeroportos ou viagens.

3. Seguindo o modelo da questão anterior, apresente um problema em que Regressão Logística é uma alternativa
a Regressão Linear e apresente (2,0):
a. Problema;
b. Dados utilizado para modelagem do problema (dataset);
c. Treinamento do modelo;
d. Resultado do modelo.
