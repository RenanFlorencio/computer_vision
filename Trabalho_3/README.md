# Tutorial de como rodar este código

1. Requisitos

Para utilizar este sistema, é preciso um dispositivo Android e um computador. Vamos começar explicando como configurar o seu computador.

## Setup do computador

### Instalação dos requisitos

É preciso instalar todos os requisitos do arquivo `requirements.txt`. Antes de instalar os requisitos, é necessário ajustar o link de download do PyTorch para a sua máquina. Para isso, acesse o site oficial do PyTorch <https://pytorch.org/get-started/locally/>, selecione as características da sua máquina e encontre o comando específico para o seu caso. Então, substitua a `--index-url` no arquivo `requirements.txt` pelo link específico. Note que os tempos de execução podem mudar dependendo das características da sua máquina e, portanto, podem não refletir o benchmark que obtivemos neste trabalho. Eles foram obtidos utilizando XXXXXXXXXX

Após isso, para instalar os requisitos, siga a sequência de comandos a seguir:

ˋˋˋ
git clone https://github.com/IgorEBatista/MC949-Visao-Computacional.git
cd MC949-Visao-Computacional/Trabalho_3
pip install -r requirements.txt
 ˋˋˋ

### TailScale

Além dos requisitos instalados pelo `requirements.txt`, é preciso garantir que o computador e o Andoid estejam conectados. Para isso, é necessário instalar uma VPN em ambos os dispositivos. Por ser gratuito e fácil de usar, escolhemos o TailScale, mas é possível utilizar outra aplicação. 

#### Download do Tailscale

Para fazer o download do TailScale, basta acessar o site oficial <https://tailscale.com/download> e fazer o download para o seu dispositivo.

#### Configurando o TailScale

Após fazer o download, abra o aplicativo tailScale no computador, e conecte em um servidor.




## Setup do Android

Para configurar seu Android, será necessário instalar alguns aplicativos: IP Webcam, TailScale e Termux. Para isso, basta abrir a loja de aplcativos e instalá-los. Agora, vamos explicar como fazer a configuração de cada um:

#### TailScale

Abra o aplicativo e conecte o seu dispositivo no mesmo servidor que o seu computador, Para isso, XXXXXXXXXXXXXXXXXXXXXXXXX


#### IP Webcam



#### Termux

Vamos começar instalando as dependências. No terminal do celular, rode os seguintes comandos:

ˋˋˋ
pkg install python
pip install flask
 ˋˋˋ

 Depois disso, basta rodar o código disponível em `src/android_api.py`. XXXXXXXXXXXXXXXXX (DESCREVER MELHOR COMO FAZER ISSO)




## Rodando o sistema

Depois de configurar o seu computador e o seu Android, basta seguir mais um passo antes de podermos rodar o sistema. Abra o arquivo `scr/assist.py` e altere o número de IP na constante IP_CELULAR_TAILSCALE para que ela seja igual ao dispositivo Android. Para verificar qual é essa constante, abra o aplicativo TailScale no computador e XXXXXXXXXXXXXXXXXXXXXXXXXX


Para isso, imediatmente depois de seguir todos aqueles passos, abra o terminal do seu computador e rode o script em `scr/assist.py`. Para isso, basta utilizar o seguinte comando:

```
cd src
python3 assist.py
```

## Calibrando o MiDaS

Para obter melhores resultados de profundidade em seu dispositivo, é possível fazer a calibração da profundidade a partir do script `src/depth.py`. Para fazer isso, siga o seguinte comando (considerando que você está dentro da pasta `cd MC949-Visao-Computacional/Trabalho_3/src`):

```
python3 depth.py
```

Ele irá abrir uma explicação de como fazer a calibração, e basta seguir as instruções dadas.