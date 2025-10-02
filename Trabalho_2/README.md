Este repositório implementa a resolução do Trabalho 2 de MC949. 

Ele possui duas seções: 

1. Resultados intermediários: implementações utilizando OpenCV que mostram resultados intermediários explícitos, os quais não conseguimos visualizar apenas com o COLMAP 
2. Pipeline do COLMAP: comandos para gerar, de fato, os modelos 3D 

O run.sh roda apenas o pipeline do COLMAP. Para gerar os resultados intermediários, é preciso rodar as célular do arquivo ```MC949_Proj2.ipynb```, as quais foram projetadas para rodar no Google Colab. Portando, para obter os resultados intermediários, basta exportar o arquivo para o Google Colab e, quando solicitado, fazer o upload de algumas imagens do cenário selecionado na pasta "fotos" do "Trabalho_2", rodando todas as células.

## Como rodar a pipeline do COLMAP

A implementação utiliza Open3d, OpenCV e COLMAP. Portanto, é necessário possuir todos instalados. Para rodar, basta clonar o repositório e, após instalar as ferramentas, rodar os seguintes comandos:

```cd Trabalho_2```
```chmod +x run.sh```
```./run.sh```