from datasets import load_dataset
import tqdm

# get the dataset only for keyphrase extraction
dataset = load_dataset("midas/kp20k", "extraction")

print("Samples for Keyphrase Extraction")

# sample from the train split
print("Sample from training data split")

# train_sample = dataset["train"][0]
for i in tqdm.tqdm(range(40000)): #(dataset["train"]):
    train_sample = dataset["train"][i]
    # Define the path where you want to save the text file.
    file_path = 'dataset/kp20k/40k.txt'

    # Write the string data to a text file.
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(" ".join(train_sample["document"]))
        # for i in range(len(train_sample["document"])):
            # file.write(train_sample["document"][i]+ " "+ train_sample["doc_bio_tags"][i]+"\n")
        file.write("\n")


