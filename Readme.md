# Interview app

This is a very simple sentiment analysis in English developed with Google's Colab.

The docker image is available in: 
> [dockerhub](https://hub.docker.com/repository/docker/gca1205/interviewapp)

Worth noting that Google's Colab is not using the latest versions of any library. Take that in mind if using my image as a source for your own project. 

To perform a prediction you will need to access through REST client applications like Insomnia or Postman. The input format is as follows: 

1. Run the app locally, through the code in my github or pulling the image from Docker. 
2. Open the Rest client application of your choice, select POST and submit a json according to the following detail:
 >{
	 "message": "i feel passionate about people particularly those i love admire and respect",
	 "method": "skl"
 }
 
>{
	 "message": "i feel passionate about people particularly those i love admire and respect",
	 "method": "dl"
 }
3. The address is: http://127.0.0.1:4000/predecir 

In this repo you will find: 
* Two trained sentiment models. 
* Two trained tokenizers files. 
* A flask api with a POST method to enable a user propose new texts for analysis. 
* A clean and vectorization class separated for clean code purposes. 
* The dockerfile using a pre-buildt Debian distribution. Alpine is still causing trouble when trying to compile with sklearn, numpy and tf. Please refer to the following (rather old) discussion [discussion](https://stackoverflow.com/questions/63163058/collecting-numpy-causes-docker-build-to-crash)
* In the folder jupyter_train I also attach the training notebooks. 

If downloading the image from Docker hub please run with the following instructions: 

> docker run -dp 4000:4000 gca1205/interviewapp

A word of caution: the app may take some time to return a reponse when using the dl method. 
