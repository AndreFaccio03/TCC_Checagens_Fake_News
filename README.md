# TCC_Checagens_Fake_News

**Trabalho de Conclusão de Curso — Processamento de Linguagem Natural: uma abordagem voltada à análise de checagem de fake news.**

- André Felipe Romero Faccio (FACOM/UFMS) | andre.faccio@ufms.br (Autor)
- Mariana Caravanti de Souza (FACOM/UFMS) | mariana.caravanti@ufms.br (Orientadora)

## Sobre o projeto

Projeto acadêmico que investiga a aplicação de técnicas de Processamento de Linguagem Natural (PLN) e aprendizado de máquina na detecção automática de fake news, utilizando textos de agências brasileiras de checagem de fatos.

A abordagem envolve a extração de informações dos textos de checagem, a geração de representações vetoriais (*embeddings*) e o treinamento de modelos de classificação.

## Resumo

A rápida disseminação de notícias falsas tornou-se um desafio crítico na era digital, exigindo o desenvolvimento de ferramentas automatizadas para auxiliar na verificação de fatos. Considerando a escassez de bases de dados rotuladas em português, este trabalho de conclusão de curso tem como objetivo avaliar modelos de linguagem de grande escala (LLMs) para desmembrar artigos publicados por agências brasileiras de checagem de fatos e extrair informações estruturadas organizadas segundo os rótulos fato e fake. Em geral, essas reportagens reúnem em um mesmo texto a alegação sob investigação e os argumentos, dados e evidências levantados pelos jornalistas para contestá-la ou confirmá-la. O desmembramento desse material permite isolar os elementos discursivos e factuais da checagem, compondo um conjunto de dados estruturado em fato e fake para treinar modelos de classificação de desinformação. Para avaliar a utilidade dos conteúdos extraídos, foram geradas representações vetoriais semânticas (dense vector embeddings) e realizados experimentos com quatro modelos de classificação: Regressão Logística, Máquinas de Vetores de Suporte (SVM), Random Forest e Multilayer Perceptron (MLP). Todo o pipeline de dados foi desenvolvido em Python, integrando o uso de LLMs executadas remotamente. Os resultados dos experimentos de classificação demonstraram que o modelo SVM obteve o melhor desempenho, alcançando 98.53% de acurácia. Os resultados apontam que o desmembramento de notícias de checagem e a extração de conteúdos sob os rótulos fato e fake constituem uma estratégia promissora para ampliar recursos em português na detecção de desinformação.

## Tecnologias utilizadas

- Python
- Pandas
- Scikit-learn
- BGE-M3
- Jupyter Notebook
- Modelos de Linguagem (LLMs)

## Fluxo de execução

Os arquivos devem ser executados na seguinte ordem:

1. `Extrator_Boato_Fatos.py` — Extração de boatos e fatos.
2. `Uniao_Datasets.ipynb` — União dos conjuntos de dados.
3. `Extracao_Embeddings.ipynb` — Extração dos embeddings textuais.
4. `Treinamento_Modelos.ipynb` — Treinamento e avaliação dos modelos de classificação.

## Modelos de classificação

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Multilayer Perceptron (MLP)

## Dados

O conjunto de dados final utilizado no projeto, para treinamento dos classificadores, está disponível no Google Drive:

- [base_final_com_vetores.parquet](https://drive.google.com/file/d/1cOpedo26vS67ltKQ-GJk8bi03DR8IEkr/view?usp=sharing)