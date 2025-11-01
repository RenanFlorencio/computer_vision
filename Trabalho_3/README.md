# Tutorial de como rodar este código

1. Requisitos

Para utilizar este sistema, é preciso um dispositivo Android e um computador. Vamos começar explicando como configurar o seu computador. Ressaltamos a importância de seguir este tutorial na ordem correta, para garantir a funcionabilidade do sistema.

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

Para fazer o download do TailScale, basta acessar o site oficial <https://tailscale.com/download> e fazer o download para o seu dispositivo. Após fazer o download, abra o aplicativo tailScale no computador, crie uma conta e conecte em sua conta.


## Setup do Android

Para configurar seu Android, será necessário instalar alguns aplicativos: DroidCam Webcam (Classic), TailScale e Termux. Para isso, basta abrir a loja de aplcativos e instalá-los. Agora, vamos explicar como fazer a configuração de cada um:

#### TailScale

Abra o aplicativo e conecte o seu dispositivo no mesmo servidor que o seu computador. Para isso, basta entrar na mesma conta que você utilizou para se conectar no computador. Ao configurar os outros aplicativos, certifique-se de que você não está fechando completantente o TailScale e nenhum outro aplicativo utilizado (deixando sempre em segundo plano).


#### DroidCam Webcam (Classic)

Para utilizar o DroidCam, basta baixar na loja de aplicativos e abri-lo. 


#### Termux

Vamos começar instalando as dependências. No terminal do celular, rode os seguintes comandos:

ˋˋˋ
pkg install python
pip install flask
pkg install vlc
 ˋˋˋ

 Depois disso, precisamos rodar o código disponível em `src/android_api.py`. Para isso, é preciso enviar esse código para o seu Android, e salvar ele em **Meus Arquivos**, com o nome `android_api.py`. Isso pode ser feito, por exemplo, enviando o arquivo por email e fazendo o download. Então, rode o seguinte comando:

ˋˋˋ
termux-setup-storage
 ˋˋˋ

Esse comando vai abrir uma página que te permite selecionar as pastas do seu celular às quais o aplicativo Termux terá acesso. Deixe selecionado apenas **Meus Arquivos** ou a pasta em que você salvou o arquivo `android_api.py`, e o próprio e **Termux**, pois são os únicos arquivos aos quais o Termux precisará de acesso para o nosso propósito.

Depois disso, volte para o terminal. No terminal, digite os comandos necessários para entrar na pasta que contém o seu arquivo `android_api.py`. Em geral, os os comandos serão os seguintes:
```
cd storage
cd downloads
```

Se esses comandos não chegarem na pasta onde está o seu arquivo `android_api.py`, você pode navegar pelos arquivos do seu celular utilizando os seguintes comandos: `ls`, que lista os arquivos da pasta atual, `cd nome_da_pasta`, o qual entra na pasta nome_da_pasta e `cd ..`, o qual volta para a pasta mãe.

Quando você já estiver na pasta que contém o seu arquivo `android_api.py`, basta rodar o comando:

```
python3 android_api.py
```
Então, deixe o código rodando em segundo plano e depois parta para os próximos passos.

## Rodando o sistema

Depois de configurar o seu computador e o seu Android, basta seguir mais um passo antes de podermos rodar o sistema. Abra o arquivo `scr/assist.py` e altere o número de IP na constante IP_CELULAR_TAILSCALE para que ela seja igual ao dispositivo Android. Para verificar qual é essa constante, abra o aplicativo TailScale no celular e verifique qual o endereço de IP do seu android. Então, substitua esse exato número na constante IP_CELULAR_TAILSCALE no arquivo `scr/assist.py`.


Imediatmente depois de seguir todos estes passos, abra o terminal do seu computador e rode o script em `scr/assist.py`. Para isso, basta utilizar os seguintes comandos:

```
cd src
python3 assist.py
```

Assim, o nosso sistema estará funcionando. Volte para o aplicativo DroidCam se quiser ver sua webcam pelo celular. A navegação por voz estará funcionando.

## Calibrando o MiDaS

Para obter melhores resultados de profundidade em seu dispositivo, é possível fazer a calibração da profundidade a partir do script `src/depth.py`. Para fazer isso, siga o seguinte comando (considerando que você está dentro da pasta `cd MC949-Visao-Computacional/Trabalho_3/src`):

```
python3 depth.py
```

Ele irá abrir uma explicação de como fazer a calibração, e basta seguir as instruções dadas.