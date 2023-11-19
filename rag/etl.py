import pandas as pd
from pathlib import Path
import numpy as np
import chromadb
from chromadb.api.models.Collection import Collection
from concurrent.futures import ThreadPoolExecutor,as_completed
from tqdm import tqdm

BLOCK_SIZE = 200
MAX_NUM = 10000

def load_car_reviews(collection: Collection,
                             folder_path: Path = Path("data/edmund_car_reviews")):
    """Prepare the car reviews dataset for ChromaDB"""

    files_path = [file_path for file_path in folder_path.iterdir()if file_path.is_file()]
    dtypes = {
        "": np.int8,
        "Review_Date": str,
        "Author_Name": str,
        "Vehicle_Title": str,
        "Review_Title": str,
        "Review": str,
        "Rating": np.float32,
    }
    # collection = chroma_client.get_or_create_collection(name="edmund_car_reviews")

    for file_path in files_path:
        filename = file_path.name
        print(f"Reading {filename}")
        car_reviews_df = pd.read_csv(file_path,lineterminator='\n', dtype=dtypes)
        if "Rating\r" in car_reviews_df.columns:
            car_reviews_df.rename(columns={"Rating\r": "Rating"}, inplace = True)
        vehicle_data = car_reviews_df["Vehicle_Title"].str.split(' ', expand=True)
        car_reviews_df["Vehicule_Date"] = vehicle_data.iloc[:,0]
        car_reviews_df["Vehicule_Brand"] = vehicle_data.iloc[:,1]
        car_reviews_df["Vehicule_Model"] = vehicle_data.iloc[:,2]
        car_reviews_df["Vehicule_Info"] = (
            vehicle_data.iloc[:,2:].stack().groupby(level=0).agg(" ".join)
        )
        # Create ids, documents, and metadatas data in the format chromadb expects
        metadata = car_reviews_df[["Review_Title", "Rating", "Vehicule_Date", 
                            "Vehicule_Model","Vehicule_Brand", "Vehicule_Info"]].to_dict(orient="records")
        reviews = car_reviews_df["Review"].to_list()
        ids = [f"{filename.split('.')[0]}_{i}" for i in range(car_reviews_df.shape[0])]
        print(f"Adding data to ChromaDB: len(ids) = {len(ids)}")
        # collection.add(documents = reviews, metadatas = metadata, ids = ids) # type: ignore
        with ThreadPoolExecutor() as executor:
            futures =[
                executor.submit(
                    collection.add,
                    documents = reviews[start:start+BLOCK_SIZE], 
                    metadatas = metadata[start:start+BLOCK_SIZE], # type: ignore
                    ids = ids[start:start+BLOCK_SIZE] 
                ) for start in np.arange(0,len(reviews),BLOCK_SIZE)
            ]
            [future.result() for future in futures]

def load_drug_reviews(collection: Collection,
                      folder_path: Path = Path("data/drugs_reviews")) -> list:
    """Prepare the car reviews dataset for ChromaDB"""

    files_path = [file_path for file_path in folder_path.iterdir()if file_path.is_file()]
    results = []
    for file_path in files_path:
        filename = file_path.name
        print(f"Reading {filename}")
        reviews_df = pd.read_csv(file_path, delimiter="\t")
        # Create ids, documents, and metadatas data in the format chromadb expects
        metadata = reviews_df[["drugName", "condition", "rating", 
                            "date","usefulCount"]].to_dict(orient="records")
        reviews = reviews_df["review"].to_list()
        ids = [f"{filename.split('.')[0]}_{i}" for i in range(reviews_df.shape[0])]
        print(f"Adding data to ChromaDB: len(ids) = {len(ids)}")
        length = min(len(reviews),MAX_NUM)
        # we're putting a min on the number of reviews, otherwise I run out of memory
        # Also, even with ThreadPool it takes several minutes just for 1000 reviews.
        pbar_length = len(np.arange(0, length, BLOCK_SIZE))
        with tqdm(total = pbar_length) as pbar:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [ 
                    executor.submit(
                        collection.add,
                        documents = reviews[start: start + BLOCK_SIZE],
                        metadatas = metadata[start: start + BLOCK_SIZE], # type: ignore
                        ids = ids[start: start + BLOCK_SIZE]
                    )
                    for start in np.arange(0, length, BLOCK_SIZE)
                ]
                for future in as_completed(futures):
                    pbar.update(1)
                    results.append(future.result())
    return results
