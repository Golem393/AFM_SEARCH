"""Matching Algorithms

Includes classes of different Matcher Algorithms

"""
import numpy as np

class MeanMatcher():
    def __init__(self, img_emb, key_emb):
        self.img_emb = img_emb
        self.key_emb = key_emb
        self.__similarity = img_emb @ key_emb.T
    def match(self, paths, k):
        means = np.mean(self.__similarity, axis=1)
        idcs = np.argsort(means)[::-1]  # Sort all indices in descending order
        
        # Create a dictionary to store the best (highest similarity) entry for each video
        seen_videos = set()
        top_paths = []
        top_similarities = []
        
        for idx in idcs:
            path = paths[idx]
            
            # Case 1: Video file (contains '.mp4_' in the path)
            if '.mp4_' in path:
                video_name = path.split('_')[0]  # Extract "video1234.mp4"
                if video_name in seen_videos:
                    continue  # Skip duplicate videos
                seen_videos.add(video_name)
            
            top_paths.append(path)
            top_similarities.append(means[idx])
            
            # Early exit if we've collected enough
            if len(top_paths) == k:
                break
        
        return top_paths, top_similarities

class ParetoFrontMatcher():
    def __init__(self, img_emb, key_emb):
        self.img_emb = img_emb
        self.key_emb = key_emb
        self.__similarity = img_emb @ key_emb.T

    def __pareto_dominance(self):
        """Calculates domination of all images to each other for Pareto Front.

        Args:
            s: np.array similarities matrix of shape (n_img, n_keys)

        Returns:
            domination_mat: np.array domination matrix
        """
        s = self.__similarity
        # Compute all possible dominations
        # Criterion 1: i dominates j if s[i][k] >= s[j][k] for all k
        crit1 = np.all(s[:,:,np.newaxis].repeat(s.shape[0], axis=-1) >= s.T, axis=1)
        # Criterion 2: i dominates j if s[i][k] > s[j][k] for at least one k
        crit2 = np.any(s[:,:,np.newaxis].repeat(s.shape[0], axis=-1) > s.T, axis=1)
        # (crit1 and crit2) = True
        domination_mat = np.all(np.array([crit1,crit2]), axis=0)

        return domination_mat
    
    def __get_pareto_front(self, mat):
        # If one column only includes False the image with the column idx is not dominated by any other image
        nondominated = np.all(mat==False, axis=0)
        # Return idx of nondominated images
        idx = np.where(nondominated)[0]
        # Set rows of nondominated values to False
        mat[idx] = False
        return idx, mat
    
    def __get_pareto_fronts(self):
        mat = self.__pareto_dominance()
        fronts = [] # collects indiced of different fronts
        idcs = [] # collects all indices
        idx, mat = self.__get_pareto_front(mat)
        fronts.append(idx)
        idcs += idx.tolist()
    
        while len(idcs) < mat.shape[0]:
            idx, mat = self.__get_pareto_front(mat)
            # check if idx already in a front
            idx = np.setdiff1d(idx, idcs)
            fronts.append(idx)
            idcs += idx.tolist()   
        return fronts
    
    def match(self, paths):
        fronts = self.__get_pareto_fronts()
        idcs = fronts[0]
        return [paths[idx] for idx in idcs]