import os
import pandas as pd
import numpy as np
from matching_algorithms import MeanMatcher

class Pipeline():
    """Driver and Setup for the image retrieval pipeline. 

    When an Pipeline instance is created, relevant models and the database containing the images 
    embeddings (if exists) is loaded into memory. The database - no matter if it exists or not -
    can be update via `update` method. The `run` method performs one retrieval search with a query.

    Attributes:
        clipmodel: Instance of a Clip Model with a get_img_embedding and get_text_embedding func.
        kwextractor: Instance of a KeywordExtractor model with an extract func.
        device: str Device on which the pipeline should run (i.e. cuda, mps, cpu)
        image_embeddings_db_path: str Path to a .h5 file or target_path where a .h5 file will be created
        
        __image_embeddings: np.array Array containing image embeddings (used to load embeddings to memory)
        __image_paths: List[str] List containing the corresponding paths to images in __image_embeddings
    """

    def __init__(self, clipmodel, kwextractor, image_embeddings_db_path: str, device: str):
        
        self.clipmodel = clipmodel
        self.kwextractor = kwextractor
        self.device = device
        self.image_embeddings_db_path = image_embeddings_db_path

        self.__image_embeddings = None
        self.__image_paths = None

        # check if image embedding database already exists and read to memory
        if os.path.isfile(image_embeddings_db_path):
            df_emb = pd.read_hdf(image_embeddings_db_path)
            self.__image_embeddings = np.array(df_emb[1].to_numpy().tolist())
            self.__image_paths = df_emb[0].tolist()

    def update(self, image_paths: list[str]):
        
        n_new_imgs = len(image_paths)
        # allocate empty array for new embeddings and their paths
        new_embeddings = np.empty(shape=(n_new_imgs, 768), dtype=np.float32) 
        new_paths_list = []

        for idx, path in enumerate(image_paths):
            emb = self.clipmodel(path, self.device)
            new_embeddings[idx] = emb.cpu().numpy()
            new_paths_list.append(path)

        # check if embedding database already exists
        if self.__image_embeddings is None:
            # no data exists yet, load new embeddings to memory
            self.__image_embeddings = new_embeddings
            self.__image_paths = new_paths_list

        else:
            # there is already data, add new embeddings to memory
            self.__image_embeddings = np.stack((self.__image_embeddings, new_embeddings), axis=0)
            self.__image_paths += new_paths_list
        
        # save embeddings in memory to disk
        self.__save_embs_to_disk(self.image_paths, self.image_embeddings)          


    def run(self, query, use_kw_extractor=False):

        # Load query embeddings
        if use_kw_extractor:
            query = self.kwextractor.extract(query, self.device)

        query_embeddings = self.clipmodel.get_txt_embedding(query)

        matcher = MeanMatcher(self.image_embeddings, query_embeddings)
        selected_imgs = matcher.match(self.image_paths)

        return selected_imgs

    def __save_embs_to_disk(self, paths_list, embeddings):
        """Private method to save image embeddings and corresponding image paths to disk.
        
        """
        df_embs = pd.DataFrame([paths_list, embeddings]).transpose()
        df_embs.to_hdf(self.image_embeddings_db_path, key="df_embs", mode="w")
