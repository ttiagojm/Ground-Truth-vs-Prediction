# Projeto Canino

* Bem, este é um projeto que eu tenho muito carinho e <b>não está finalizado!</b>

* As versões v1 e v2, são puros testes, onde eu *aprendi*, *diverti-me* e *trabalhei por horas*.


* O código está documentado, dei o meu melhor durante o desenvolvimento. 

# O que temos aqui?

A versão 1 é uma versão mais "escrita na unha" e a versão 2 é mais compacta e talvez até funcione melhor.

Resumindo:

* Vão ter possibilidade de fazer donwload de todas as imagens, de cada raça, disponíveis no ImageNet, tudo de forma automática (~7GB)

* Serão gerados logs de métricas e histogramas tanto de métricas como dos gradientes de treino, que podem ser visualizado no <b>Tensorboard</b>

* Na pasta `analysis` vão ter:
	
	* O .csv com os urls de download das *synnets* do ImageNet, assim como features criadas por mim, para cada raça. <b>Ver Disclaimer abaixo</b>

	* Aqui analisa-se um pouco as raças calculando cossenos de similaridade e criação de dendogramas. Isto baseado nas features que criei e nos pesos que defini.

* Copiei a implementação do InceptionV3 do Keras e dei uma pequena limpeza, para podermos fazer alterações em hiperparâmetros, mais facilmente.

* Data Augmentation com OpenCV

# DISCLAIMER

[1] - As imagens não são tratadas pela ImageNet, para isso precisamos pedir permissão e isso requer pretencermos a uma instituição ou então baixar por outras fontes não tão fidedignas. 

<b>Dito isto, há imagens que podem conter software malicioso embutido!! Eu mesmo apanhei uma.</b> O meu conselho é usar o Google Colab e fazer o download lá e descarregar para a sua máquina. Verifiquei que eles removem a grande maioria das imagens que podem conter algo malicioso.

[2] - As features criadas por mim, foram feitas manualmente, olhando as características das raças neste [site](https://www.dogbreedinfo.com/abc.htm), portanto se encontrar nomes engraçados é porque eu não sabia como classificar o pêlo de uma dada raça.

[3] - Todas as imagens são obtidas através da [ImageNet](https://www.image-net.org/), e como não são tratadas, vêm muitos ícones de sites e imagens corrompiadas (mesmo tendo funções para verificar e elminar essas imagens).<b>É importante que você vejas as imagens e elimine e edite o que achar conveniente</b>

[4] - Poderá haver caminhos para ficheiros e/ou diretórios errados, já que tive de mexer na estrutura para criar este repositório.


# Porque na versão 1 não usou funções prontas do TF/Keras?
Inicialmente utilizei, mas eu queria obter os gradientes e visualizá-los e como o Tensorflow removeu um parâmetro que fazia isso automaticamente, eu tive que refazer tudo na mão para poder criar os histogramas dos gradientes ... Sim eu fiz tudo isso por um gráfico bonito.


# O projeto morreu?
Eu ainda tentei recriar ele com PyTorch (versão 3) e quem sabe no futuro, em Julia (versão 4 e final), mas eu desanimei do projeto ... Então é provável que fique parado por uns tempos.