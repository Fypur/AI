#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

typedef struct network
{
    int nb_layers;
    int* layers;

    float** neurons;
    float** biases;
    float*** weights;
    float** z;

    float learningRate;
} network;

float rand_float(){
    return (float)rand() / RAND_MAX;
}

float sigmoid(float x){
    return 1 / (1 + (float)exp(-x));
}

float sigmoid_prime(float x){
    return sigmoid(x) * (1 - sigmoid(x));
}

network* create_network(int nb_layers, int* layers){
    srand(time(NULL));

    network* net = malloc(sizeof(network));

    net->nb_layers = nb_layers;

    net->layers = malloc(sizeof(int) * nb_layers);
    for(int i = 0; i < nb_layers; i++)
        net->layers[i] = layers[i];
    
    net->learningRate = 0.01;

    net->neurons = malloc(sizeof(float*) * nb_layers);
    for(int i = 0; i < nb_layers; i++){
        net->neurons[i] = malloc(sizeof(float) * layers[i]);

        for(int j = 0; j < layers[i]; j++){
            net->neurons[i][j] = 0;
        }
    }

    net->biases = malloc(sizeof(float*) * nb_layers);
    for(int i = 1; i < nb_layers; i++){ //on part de i = 1 car la première layer (celle de l'input) n'a pas de biais
        net->biases[i] = malloc(sizeof(float) * layers[i]);

        for(int j = 0; j < layers[i]; j++){
            net->biases[i][j] = 2;//rand_float();
        }
    }

    net->z = malloc(sizeof(float*) * nb_layers);
    for(int i = 1; i < nb_layers; i++){ //i = 1 pour la même raison
        net->z[i] = malloc(sizeof(float) * layers[i]);

        for(int j = 0; j < layers[i]; j++){
            net->z[i][j] = 0;
        }
    }

    net->weights = malloc(sizeof(float**) * nb_layers);
    for(int i = 1; i < nb_layers; i++){ //on part de i = 1 pour la même raison
        net->weights[i] = malloc(sizeof(float*) * layers[i]);
        //weights[i][j][k] représente le poids arrivant au neurone j de la layer i, partant du neurone k de la layer i - 1

        for(int j = 0; j < layers[i]; j++){
            net->weights[i][j] = malloc(sizeof(float) * layers[i - 1]);

            for(int k = 0; k < layers[i - 1]; k++){
                net->weights[i][j][k] = 1;//rand_float();
            }
        }
    }

    return net;
}

float* feedforward(network* net, float* input){
    for(int i = 0; i < net->layers[0]; i++){ //on copie l'input dans la première layer
        net->neurons[0][i] = input[i];
    }

    //feed forward
    for(int i = 1 ; i < net->nb_layers; i++){ //pour chaque layer
        
        for(int j = 0; j < net->layers[i]; j++){ //pour chaque neurone dans cette layer

            net->z[i][j] = 0;

            for(int k = 0; k < net->layers[i - 1]; k++){ //pour chaque neurone dans la layer précédente
                net->z[i][j] += net->weights[i][j][k] * net->neurons[i - 1][k];
            }

            net->z[i][j] += net->biases[i][j];

            net->neurons[i][j] = sigmoid(net->z[i][j]); //sigmoid
        }
    }

    //copier resultat depuis dernière layer des neurones
    float* result = malloc(sizeof(float) * net->layers[net->nb_layers - 1]);
    for(int i = 0; i < net->layers[net->nb_layers - 1]; i++){
        result[i] = net->neurons[net->nb_layers - 1][i];
    }

    return result;
}

int main(){
    int layers[] = { 3, 3, 3 };
    network* net = create_network(3, layers);
    float input[] = { 0, 0, 0 };
    float* output = feedforward(net, input);

    for(int i = 0; i < 3; i++) printf("%f\n", output[i]);

    return 0;
}