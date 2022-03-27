import numpy as np
from match_cost_function import sad_similarity_measure

class StereoBlockMatcher:
    # implementation based on my solution for CS 4476 course and
    # article - mccormickml.com/assets/StereoVision/Stereo%20Vision%20-%20Mathworks%20Example%20Article.pdf

    def __init__(self, block_size, max_search_range, sim_func=sad_similarity_measure, max_disparity=3):
        self.block_size = block_size
        self.max_search_range = max_search_range
        self.sim_func = sim_func
        self.max_disparity = max_disparity
        pass


    def sub_pixel_refine(self, d, c1, c2, c3):
        np_c = np.array([c1, c2, c3])
        max_c = np.max(np_c)
        np_c = np_c / max_c
        c1, c2, c3 = np_c
        return d - (0.5 * (c3 - c1) / (c1 - (2 * c2) + c3))


    def compute_subpixel_disparity(self, np_disparities, best_disp):
        idx1 = len(np_disparities)-1 if best_disp == 0 else best_disp-1
        idx3 = 0 if best_disp == len(np_disparities)-1 else best_disp+1
        c1, c2, c3 = np_disparities[idx1], np_disparities[best_disp], np_disparities[idx3]
        disparity = self.sub_pixel_refine(best_disp, c1, c2, c3)
        return disparity


    def stereo_compute(self, im_left, im_right, default_cost=255, use_subpixel_refine=False):

        if (im_left.shape != im_right.shape):
            raise Exception('Error: shape of images no equals...')

        H, W, _ = im_left.shape
        h_b = self.block_size // 2
        
        disparity_map = np.zeros((H - 2*h_b, W-2*h_b))

        for y in range(h_b, H-h_b):
            for x in range(h_b, W-h_b):
                patch1 = im_left[y-h_b:y+h_b+1, x-h_b:x+h_b+1]
                offset = 0
                disparities = []
                while (offset < self.max_search_range and x-offset-h_b >= h_b):
                    patch2 = im_right[y-h_b:y+h_b+1, x-h_b-offset:x+h_b-offset+1] 
                    sim_value = self.sim_func(patch1, patch2)
                    offset += 1
                    disparities.append(sim_value)
              
                if len(disparities) < self.max_disparity:
                    disparities += [default_cost] * (self.max_disparity - len(disparities))

                np_disparities = np.array(disparities)
                disparity = np.argmin(np_disparities)

                if use_subpixel_refine:
                    if disparity != 0 and disparity != self.max_search_range-1:
                        disparity = self.compute_subpixel_disparity(np_disparities, disparity)

                disparity_map[y-h_b, x-h_b] = disparity
                         
        return disparity_map


    def stereo_dynamic_compute(self, im_left, im_right, disparity_penalty=0.65):
        
        if (im_left.shape != im_right.shape):
            raise Exception('Error: shape of images no equals...')

        H, W, _ = im_left.shape
        h_b = self.block_size // 2
        
        disparity_map = np.zeros((H, W))
        disparity_cost = np.ones((W, 2 * self.max_search_range + 1)) 
        
        for y in range(0, H):
            disparity_cost[:] = np.Inf
            rowmin_bl = max(0, y - h_b) # row min block
            rowmax_bl = min(H, y + h_b) # row max block
            for x in range(0, W):
                colmin_bl = max(0, x - h_b)
                colmax_bl = min(W, x + h_b)

                offsetmin_block = max(-self.max_search_range, 1 - colmin_bl)
                offsetmax_block = min(self.max_search_range, W-1-colmax_bl)

                patch1 = im_left[rowmin_bl:rowmax_bl, colmin_bl:colmax_bl]
                offset = offsetmin_block
                while (offset < offsetmax_block):
                    patch2 = im_right[rowmin_bl:rowmax_bl, colmin_bl+offset:colmax_bl+offset] 
                    sim_value = self.sim_func(patch1, patch2)
                    offset += 1
                    disparity_cost[x, offset + self.max_search_range] = sim_value

            optim_ids = np.zeros(disparity_cost.shape)
            cost_path = disparity_cost[x, :]
            end = cost_path.shape[0]
            
            for i in range(W-1, -1, -1):
                # Find the minimum value in each column of this matrix
                search_block = np.array([ [np.Inf, np.Inf] + (cost_path[0:end-4] + disparity_penalty*3).tolist(),
                                          [np.Inf] + (cost_path[0:end-3] + disparity_penalty*2).tolist(),
                                          (cost_path[0:end-2] + disparity_penalty).tolist(),
                                          cost_path[1:end-1].tolist(),
                                          (cost_path[2:end] + disparity_penalty).tolist(),
                                          (cost_path[3:end] + disparity_penalty*2).tolist() + [np.Inf],
                                          (cost_path[4:end] + disparity_penalty*3).tolist() + [np.Inf, np.Inf] ])
                
                min_v, min_ids = np.min(search_block, axis=0), np.argmin(search_block, axis=0)
                
                select_pixels = disparity_cost[i, 1:-1] + min_v
                
                cost_path = np.array([np.Inf] + select_pixels.tolist() + [np.Inf])
                
                select_ids = np.array(range(1, disparity_cost.shape[1]-1))
                
                optim_ids[i, 1:-1] = select_ids + (min_ids - 3)
            
            min_cost = np.argmin(cost_path)
            disparity_map[y, 0] = min_cost
            for j in range(0, W-1):
                pix_lut = np.int32(max(0, min(optim_ids.shape[1], np.round(disparity_map[y, j]))))
                disparity_map[y, j+1] = optim_ids[j, pix_lut]
                         
        return disparity_map