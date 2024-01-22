import os
from chainer import serializers
from utility.vae import VAE
from utility.predict import predict
from tqdm import tqdm
from utility.txt2xml import txt2xml

def generate(latent_vecs = [None], repeat_times = 1, file_path=".", model_dir="saved_models"):
    #   Construct vae
    vae = VAE(n_embed = 50, n_layers = 1, hidden_size = 400, drop_ratio = 0.3, n_latent = 60, batch_size = 20, train_file = f"{file_path}/{model_dir}/train.txt", epoch = 501)
    # vae.to_gpu()
    model_name = f"{file_path}/{model_dir}/models/aibirds_word_500"
    serializers.load_npz(model_name, vae)
    #   txt to xml
    deconverter = txt2xml()
    os.makedirs(f"{file_path}/generated_levels", exist_ok=True)
    
    #   Generate for each latent vector
    for i in tqdm(range(len(latent_vecs))):
        for j in range(repeat_times):
            # Predict sentences
            tenti = predict(vae, 1, z = latent_vecs[i])
            vae.dec.reset_state()
            text, pig_count, tnt_count = deconverter.vector2xml(tenti[0])
            # Write level
            with open(f"{file_path}/generated_levels/level-vec-{i}-{j}.xml", "w") as f:
                f.write(text)
            print(f"\npig count: {pig_count}, tnt count: {tnt_count}")

def fitness(latent_vecs = [None], file_path=".", model_dir="saved_models"):
    #   Construct vae
    vae = VAE(n_embed = 50, n_layers = 1, hidden_size = 400, drop_ratio = 0.3, n_latent = 60, batch_size = 20, train_file = f"{file_path}/{model_dir}/train.txt", epoch = 501)
    # vae.to_gpu()
    model_name = f"{file_path}/{model_dir}/models/aibirds_word_500"
    serializers.load_npz(model_name, vae)
    fitness_list = []
    deconverter = txt2xml()
    # For each latent vector(level)
    for i in tqdm(range(len(latent_vecs))):
        # Predict sentences
        tenti = predict(vae, 1, z = latent_vecs[i])
        vae.dec.reset_state()
        pig_count, tnt_count, material_types, object_types, object_count = deconverter.fitness_cal(tenti[0])
        fitness_list.append([pig_count, tnt_count, material_types, object_types, object_count])

    return fitness_list

if __name__ == "__main__":
    generate()
