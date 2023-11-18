import pandas as pd
from pathlib import Path
import numpy as np
import chromadb
from chromadb.api import ClientAPI

def prepare_car_reviews_data(chroma_client: ClientAPI,
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
    collection = chroma_client.get_or_create_collection(name="edmund_car_reviews")

    for file_path in files_path:
        filename = file_path.name
        print(f"Reading {filename}")
        car_reviews_df = pd.read_csv(files_path[0],lineterminator='\n', dtype=dtypes)
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
        collection.add(documents = reviews, metadatas = metadata, ids = ids) # type: ignore


if __name__=="__main__":
    print("Creating Chroma Client")
    chroma_client = chromadb.HttpClient(host='localhost', port=8000) # type: ignore
    prepare_car_reviews_data(chroma_client)