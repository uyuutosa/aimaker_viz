import cv2
from numpy import *
import PIL.Image as I
import pandas as pd
import numpy as np
import aimaker.predictor.segmentation_predictor as sp
import subprocess as sb

class LucasEstimator:
    def __init__(self, path, height, weight):
        print("php {} {} {}".format(path, height, weight))
        popen = sb.Popen("php {} {} {}".format(path, height, weight).split(), stdout=sb.PIPE)
        line = popen.communicate()[0]
        self.line = line
        lst  = [{x[0].strip(b'"') : float(x[1].strip(b'"'))} for x in [x.split(b':') for x in line[2:].strip(b'{}').split(b',')]]
        self.param_dic = {}
        [self.param_dic.update(x) for x in lst]                                                                              

    def __getitem__(self, key):
        return self.param_dic[key]
        

class TrueCondition:
    def __init__(self):
        pass

    def __call__(self, p1, p2, p3, p4):
        return True

class AngleBoundaryCondition:
    def __init__(self, min_angle, max_angle, vec):
        self.min_angle, self.max_angle = min_angle, max_angle
        self.vec = array(vec)

    def __call__(self, p1, p2, p3, p4):
        x2,y2 = p3 - p4
        theta = abs((angle(x2 + y2 * 1.j) - angle(self.vec[0] + self.vec[1] * 1.j))  / pi * 180)
        #print("angle is {}".format(theta))
        return theta > self.min_angle and theta <  self.max_angle
        
        
class clothingSizeEstimator:
    def __init__(self, 
                 frontal_image_path, 
                 side_image_path, 
                 frontal_raw_image_path,
                 side_raw_image_path,
                 height_cm=175,
                 weight_kg=65,
                 lucas_path="./lucas.php",
                 feel='normal',
                 bicep_critical_value=(0,40)):

        self.frontal_image_path = frontal_image_path
        self.side_image_path    = side_image_path
        frontal_image = I.open(frontal_image_path)
        self.frontal_resized_arr = array(frontal_image)

        side_image = I.open(side_image_path)
        self.side_resized_arr = array(side_image)

        frontal_raw_image = I.open(frontal_raw_image_path)
        self.frontal_raw_arr = array(frontal_raw_image)

        side_raw_image = I.open(side_raw_image_path)
        self.side_raw_arr = array(side_raw_image)

        self.frontal_resize_ratio = self.frontal_raw_arr.shape[0] / self.frontal_resized_arr.shape[0]
        self.side_resize_ratio    = self.side_raw_arr.shape[0]    / self.side_resized_arr.shape[0]

        self.height_cm = height_cm
        self.weight_kg = weight_kg
        self.lucas = LucasEstimator(lucas_path, height_cm, weight_kg)
        self.feel = feel
        self.bicep_critical_value = bicep_critical_value

    def _getCorrectValue(self):
        return {'tight':0.1,'normal':0.2,'loose':0.3}[self.feel]

    def _correctIfLucasIsMoreProper(self, result_dic, point_dic, key=None, lucas_key=None):
        if key is not None:
            if result_dic[key] > self.lucas[lucas_key]:
                result_dic[key] = self.lucas[lucas_key]
        return result_dic, point_dic
                

    def getExtractBackgroundImages(self,
                                   transform='', 
                                   gpu_id=0, 
                                   divide_size=(1,1), 
                                   dump_path='tmp',
                                   pad=40,
                                   thresh=5):

        self.frontal_resized_extracted_arr, self.frontal_raw_extracted_arr = self._extractBackgroundOfHuman(\
                                   self.frontal_raw_arr, 
                                   self.frontal_resize_ratio,
                                   transform=transform, 
                                   gpu_id=gpu_id, 
                                   divide_size=divide_size, 
                                   dump_path=dump_path,
                                   image_path=self.frontal_image_path,
                                   )

        self.side_resized_extracted_arr, self.side_raw_extracted_arr = self._extractBackgroundOfHuman(\
                                   self.side_raw_arr, 
                                   self.side_resize_ratio,
                                   transform=transform, 
                                   gpu_id=gpu_id, 
                                   divide_size=divide_size, 
                                   dump_path=dump_path,
                                   image_path=self.side_image_path)


        self.frontal_binary = self._getBinaryImage(self.frontal_resized_extracted_arr, thresh=thresh)
        self.side_binary    = self._getBinaryImage(self.side_resized_extracted_arr, thresh=thresh)

        self.frontal_raw_trimed_arr, self.frontal_offset_info = self._getHumanAroundImage(\
                                                self.frontal_binary, 
                                                self.frontal_raw_arr, 
                                                pad=pad,
                                                resize_ratio=self.frontal_resize_ratio)

        self.side_raw_trimed_arr, self.side_offset_info = self._getHumanAroundImage(\
                                                self.side_binary, 
                                                self.side_raw_arr, 
                                                pad=pad,
                                                resize_ratio=self.side_resize_ratio)

        #print(self.side_resized_extracted_arr.shape, self.side_raw_extracted_arr.shape)
        #self.frontal_trimed_resize_ratio = self.frontal_raw_trimed_arr.shape[0] / 512
        #self.side_trimed_resize_ratio    = self.side_raw_trimed_arr.shape[0]    / 512
        #print(self.side_binary.shape)
        #print("hello")

        #print(self.frontal_raw_trimed_arr.shape)
        #print(self.side_raw_trimed_arr.shape)
        #print("alsdkflj")
        #input("ahaha")
        self.frontal_resized_trimed_extracted_arr, self.frontal_raw_trimed_extracted_arr = self._extractBackgroundOfHuman(\
                                   self.frontal_raw_trimed_arr, 
                                   self.frontal_resize_ratio, 
                                   #self.frontal_trimed_resize_ratio, 
                                   transform=transform, 
                                   gpu_id=gpu_id, 
                                   divide_size=divide_size, 
                                   dump_path=dump_path,
                                   image_path=self.frontal_image_path,
                                   )

        self.side_resized_trimed_extracted_arr, self.side_raw_trimed_extracted_arr = self._extractBackgroundOfHuman(\
                                   self.side_raw_trimed_arr, 
                                   self.side_resize_ratio, 
                                   #self.side_trimed_resize_ratio, 
                                   transform=transform, 
                                   gpu_id=gpu_id, 
                                   divide_size=divide_size, 
                                   dump_path=dump_path,
                                   image_path=self.side_image_path)

        self.frontal_binary = self._getBinaryImage(self.frontal_raw_trimed_extracted_arr, thresh=thresh)
        self.side_binary    = self._getBinaryImage(self.side_raw_trimed_extracted_arr, thresh=thresh)

        self.frontal_raw_outlined_arr, self.frontal_contour = self._drawOutline(\
                                                      self.frontal_binary,
                                                      self.frontal_raw_arr,
                                                      self.frontal_offset_info,
                                                      self.frontal_resize_ratio)

        self.side_raw_outlined_arr, self.side_contour = self._drawOutline(\
                                                      self.side_binary,
                                                      self.side_raw_arr,
                                                      self.side_offset_info,
                                                      self.side_resize_ratio)


    def getPoseImages(self, dump_path=None, gpu_id=0, weight_name='./model/pose_model.pth'):
        frontal_axis_info, self.frontal_pose_labeled_image = self._estimatePose(\
                                                          self.frontal_resized_trimed_extracted_arr, 
                                                          self.frontal_raw_trimed_extracted_arr, 
                                                          self.frontal_resize_ratio, 
                                                          #self.frontal_trimed_resize_ratio, 
                                                          dump_path=dump_path, 
                                                          gpu_id=gpu_id,
                                                          weight_name=weight_name
                                                          )

        side_axis_info, self.side_pose_labeled_image = self._estimatePose(\
                                                          self.side_resized_trimed_extracted_arr, 
                                                          self.side_raw_trimed_extracted_arr, 
                                                          self.side_resize_ratio, 
                                                          #self.side_trimed_resize_ratio, 
                                                          dump_path=dump_path, 
                                                          gpu_id=gpu_id,
                                                          weight_name=weight_name)

        self.frontal_p_dic, self.side_p_dic = self._parsePoints(frontal_axis_info, side_axis_info)

    def getImage(self):
        self.frontal_ratio  = self._getRatio(self.height_cm, self.frontal_binary)
        self.side_ratio     = self._getRatio(self.height_cm, self.side_binary)

        c = self._correctIfLucasIsMoreProper

        self._initializeLabeledImage()
        result_dic = {}
        point_dic = {}
        result_dic, point_dic = c(*self.estimateNeck(result_dic, point_dic))#, 'neck_circumference', b'neck')
        result_dic, point_dic = c(*self.estimateShoulderWidth(result_dic, point_dic))#, 'shoulder_width', b'shoulder')
        result_dic, point_dic = c(*self.estimateForeArm(result_dic, point_dic))
        result_dic, point_dic = c(*self.estimateWaist(result_dic, point_dic))
        result_dic, point_dic = c(*self.estimateChestWidth(result_dic, point_dic))#, 'chest_circumference', b'chest')
        result_dic, point_dic = c(*self.estimateHem(result_dic, point_dic))
        result_dic, point_dic = c(*self.estimateWrist(result_dic, point_dic))
        result_dic, point_dic = c(*self.estimateSleeve(result_dic, point_dic))
        result_dic, point_dic = c(*self.estimateBicep(result_dic, point_dic))
        result_dic, point_dic = c(*self.estimateThigh(result_dic, point_dic))
        result_dic, point_dic = c(*self.estimateCalf(result_dic, point_dic))
        result_dic, point_dic = c(*self.estimateAnkle(result_dic, point_dic))
        result_dic, point_dic = c(*self.estimateTrouserLength(result_dic, point_dic))
        result_dic, point_dic = c(*self.estimateBodyLength(result_dic, point_dic))
        return result_dic

    def _initializeLabeledImage(self):
        self.frontal_labeled_arr = self.frontal_raw_trimed_arr.copy()[...,::-1].astype(np.uint8)
        self.side_labeled_arr    = self.side_raw_trimed_arr.copy()[...,::-1].astype(np.uint8)


    def _parsePoints(self, 
                     frontal_axis_info,
                     side_axis_info,
                    ):

        self.auau = frontal_axis_info
        self.au = side_axis_info
        frontal_points = pd.DataFrame(frontal_axis_info)
        frontal_points[frontal_points == 17] = 16
        frontal_points = frontal_points.set_index(0).groupby(0).mean().astype(int)
        side_points    = pd.DataFrame(side_axis_info)#.set_index(0).groupby(0).mean().astype(int)
        side_points[side_points == 17] = 16
        side_points = side_points.set_index(0).groupby(0).mean().astype(int)
        #frontal_points = pd.read_csv(frontal_axis_info_path, header=None).set_index(0).groupby(0).mean().astype(int)
        #side_points    = pd.read_csv(side_axis_info_path, header=None).set_index(0).groupby(0).mean().astype(int)

        ## for froontal images
        # point for calculate normal vector
        class ratioCrrectiveDic:
            def __init__(self, p_dic, ratio):
                self.p_dic = p_dic
                self.ratio = ratio

            def update(self, dic):
                self.p_dic.update(dic)


            def __setitem__(self, k, v):
                self.p_dic[k] = v

            def __getitem__(self, k):
                return (self.p_dic[k][0] * self.ratio).astype(int), (self.p_dic[k][1] * self.ratio).astype(int)
            
        frontal_p_dic = ratioCrrectiveDic({}, self.frontal_resize_ratio)
        frontal_p_dic['left_bicep']      = self._calcIntermedPoints(frontal_points, 5, 6)
        frontal_p_dic['right_bicep']     = self._calcIntermedPoints(frontal_points, 2, 3)
        frontal_p_dic['left_fore_arm']   = self._calcIntermedPoints(frontal_points, 6, 7)
        frontal_p_dic['right_fore_arm']  = self._calcIntermedPoints(frontal_points, 3, 4)
        frontal_p_dic['left_wrist']     = array(frontal_points.ix[array([7,   6])])
        frontal_p_dic['right_wrist']      = array(frontal_points.ix[array([4,   3])])
        frontal_p_dic['left_thigh']      = self._calcIntermedPoints(frontal_points, 11, 12, 0.8)
        frontal_p_dic['right_thigh']     = self._calcIntermedPoints(frontal_points,  8,  9, 0.8)
        frontal_p_dic['left_calf']       = self._calcIntermedPoints(frontal_points, 12, 13)
        frontal_p_dic['right_calf']      = self._calcIntermedPoints(frontal_points,  9, 10)
        frontal_p_dic['left_ankle']      = array(frontal_points.ix[array([13,   12])])
        frontal_p_dic['right_ankle']     = array(frontal_points.ix[array([10,   9])])

        # point for calculate tangent vector
        frontal_p_dic['shoulder'] = self._calcShoulderPoints(frontal_points)#array(frontal_points.ix[array([2, 5])])
        frontal_p_dic['hem']      = array(frontal_points.ix[array([8, 11])])
        frontal_p_dic['chest']    = self._calcChestPoints(frontal_points)
        frontal_p_dic['waist']    = self._calcWaistPoints(frontal_points)
        frontal_p_dic['neck_width']    = self._calcNeckPoints(frontal_points)

        ## for side images
        # point for calculate tangent vector
        side_p_dic = {}
        side_p_dic = ratioCrrectiveDic({}, self.side_resize_ratio)
        side_p_dic['shoulder'] = self._calcShoulderPoints(frontal_points)#array(side_points.ix[array([2, 5])]) #to reviseion
        side_p_dic['hem']      = self._calcHemPoints(side_points)
        side_p_dic['chest']    = self._calcChestPoints(side_points)
        side_p_dic['waist']    = self._calcWaistPoints(side_points)
        side_p_dic['neck_width']    = self._calcSideNeckPoints(side_points)

        self.frontal_points = frontal_points

        return frontal_p_dic, side_p_dic


    def _calcIntermedPoints(self, points, i, j, ratio=0.5):
        dist = linalg.norm(points.ix[i] - points.ix[j])
        t = (points.ix[j] - points.ix[i])/ dist
        intermed = array((points.ix[i] + points.ix[j]) / 2)
        return concatenate(((array(points.ix[i]) + ratio * dist * t)[None,:] , array(points.ix[j])[None,:]), axis=0)
    #def _calcIntermedPoints(self, points, i, j, ratio=0.5):
    #    points.ix[i] + points.ix[j]
    #    intermed = array((points.ix[i] + points.ix[j]) / 2)
    #    return concatenate((intermed[None,:], array(points.ix[j])[None,:]), axis=0)


    def _drawOutline(self, binary, raw, offset_info, resize_ratio):
        raw = raw.copy()
        imgEdge, contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        lst = [len(x) for x in contours]
        cnt = contours[argmax(lst)].copy()
        cnt_for_draw = contours[argmax(lst)]
        cnt_for_draw[:,0,0] += offset_info["x"]
        cnt_for_draw[:,0,1] += offset_info["y"]



        ##savetxt("value.txt", cnt[:, 0, :].T)

        return cv2.drawContours(raw.copy(),[(cnt_for_draw).astype(int32)] , -1, (0,255,0), 6).astype(uint8), cnt
        #wreturn raw, cnt

    def _calcNeckPoints(self, points):
        t = (array(points.ix[0]) - points.ix[1])[None,:]* 1.

        distance = linalg.norm(t) 
        
        t /= distance
        return concatenate((array(points.ix[1]) + distance /4 * t, array(points.ix[0])[None,:]), axis=0)

    def _calcSideNeckPoints(self, points):
        t = (array(points.ix[16]) - points.ix[1])[None,:]* 1.

        distance = linalg.norm(t) 
        
        t /= distance

        return concatenate((array(points.ix[1]) + distance / 4 * t, array(points.ix[16])[None,:]), axis=0)


    def _calcWaistPoints(self, points):
        center_hip_point = array((points.ix[8] + points.ix[11]) / 2)
        t = (array(points.ix[1]) - center_hip_point)[None,:]
        distance = linalg.norm(t)
        
        t /= distance
        return concatenate((center_hip_point + distance / 3 * t, array(points.ix[1])[None,:]), axis=0)

    def _calcShoulderPoints(self, points):
        center_hip_point = array((points.ix[8] + points.ix[11]) / 2)
        t = (array(points.ix[1]) - center_hip_point)[None,:]
        distance = linalg.norm(t)
        
        t /= distance
        return concatenate((center_hip_point + distance * 21 / 20 * t, array(points.ix[1])[None,:]), axis=0)

    def _calcChestPoints(self, points):
        center_hip_point = array((points.ix[8] + points.ix[11]) / 2)
        t = (array(points.ix[1]) - center_hip_point)[None,:]
        distance = linalg.norm(t)
        
        t /= distance
        return concatenate((center_hip_point + distance * 3 / 4 * t, array(points.ix[1])[None,:]), axis=0)

    def _calcHemPoints(self, points):
        center_hip_point = array((points.ix[8] + points.ix[11]) / 2)
        return concatenate((center_hip_point[None, :], array(points.ix[1])[None,:]), axis=0)

    def _getBinaryImage(self, arr, thresh=0):
        kernel = ones((5,5),np.uint8)
        return cv2.morphologyEx(where(arr.max(axis=2)>thresh, 255, 0).astype(uint8), cv2.MORPH_OPEN, kernel)
       # return #.max(axis=2)

    def _getHumanAroundImage(self, binary, raw, pad, resize_ratio):
        size = binary.max(axis=1).size
        for n,i in enumerate(binary.max(axis=1)):
            if i:
                break
        top = n
        for n,i in enumerate(binary.max(axis=1)[::-1]):
            if i:
                break
        bottom = n

        size = binary.max(axis=0).size
        for n,i in enumerate(binary.max(axis=0)):
            if i:
                break
        left = n
        for n,i in enumerate(binary.max(axis=0)[::-1]):
            if i:
                break
        right = n
        #print(top,bottom,left,right)
        #print(binary[top-pad:-1-bottom+pad, left-pad:-1-right+pad])
        i = top-pad if top-pad >= 0 else 0
        j = -1-bottom+pad if -1-bottom+pad < 0 else -1
        k = left-pad if left-pad >= 0 else 0
        l = -1-right+pad if -1-right+pad < 0 else -1

        i = int(i * resize_ratio)
        j = int(j * resize_ratio) 
        if j == 0:
            j = -1
        k = int(k * resize_ratio)
        l = int(l * resize_ratio)
        if l == 0:
            l = -1



        return raw[i:j, k:l],\
               {'y':i, 'x':k}
        #return \#binary[top-pad:-bottom+pad, left-pad:-1-right+pad],\
    
    def _getRatio(self, height_cm, binary):
        size = binary.max(axis=1).size
        for n,i in enumerate(binary.max(axis=1)):
            if i:
                break
        top = n
        for n,i in enumerate(binary.max(axis=1)[::-1]):
            if i:
                break
        bottom = n
        
        ratio = height_cm / ((size - bottom) - top)
        #cv2.circle(binary, (100, int(top)), 7, 255, -1)
        #cv2.circle(binary, (100, int(size - bottom)), 7, 255, -1)
        return ratio

    def _calcTangentDistance(self, 
                             points, 
                             arr, 
                             binary, 
                             ratio, 
                             labeled_arr, 
                             max_length=1000, 
                             name=None,
                             n_offset=(1,1),
                             correction_factor=0.0,
                             limit_boundary=None
                             ):
        x_lst = []
        y_lst = []
        x2_lst = []
        y2_lst = []
        t_lst = []
        length_lst = []
        ini = points[0]
        for offset_x in range(-n_offset[1], n_offset[1], 1):
            for offset_y in range(-n_offset[0], n_offset[0], 1):
                t = points[1] - ini 
                t = t  * 1.
                t /= linalg.norm(t)
                for i in range(max_length):
                    x,y = (ini + i*t).astype(int)
                    if not binary[y, x]:
                        break
                #I.fromarray(arr)
                
                for i in range(max_length):
                    x2,y2 = (ini + i*(-t)).astype(int)
                
                    if not binary[y2, x2]:
                        break
                
                length = linalg.norm(array([x2,y2])-array([x,y])) * ratio
                length_lst   += [length]
                x_lst += [x]
                y_lst += [y]
                x2_lst += [x2]
                y2_lst += [y2]
                t_lst += [t]


        length_arr = array(length_lst)
        x_arr = array(x_lst)
        y_arr = array(y_lst)
        x2_arr = array(x2_lst)
        y2_arr = array(y2_lst)
        t_arr = array(t_lst)
        
        length = length_arr.min()
        x2 = x2_arr[length_arr.argmin()]
        y2 = y2_arr[length_arr.argmin()]
        x = x_arr[length_arr.argmin()]
        y = y_arr[length_arr.argmin()]
        t = t_arr[length_arr.argmin()]


        if limit_boundary is not None:
            p1, p2 = array([x,y]), array([x2,y2])
            point = self._getIntersection(p1, p2, *limit_boundary)
            dist1 = linalg.norm(p1 - point)
            dist2 = linalg.norm(p2 - point)
            dist3 = linalg.norm(p2 - p1)
            if dist1 < dist2:
                if dist3 > dist1:
                    x2,y2 = point
            else:
                if dist3 > dist2:
                    x,y = point
            #cv2.circle(labeled_arr, tuple(limit_boundary[0]), 7, (0, 0, 255), -1)
            #cv2.circle(labeled_arr, tuple(limit_boundary[1]), 7, (0, 255, 0), -1)

        distance = linalg.norm(array([x2,y2]) - array([x,y])) 
        length = distance * ratio * (1 - correction_factor)


        loc = tuple(((array([x, y]) - t * distance * correction_factor/2)).astype(int))
        loc2  = tuple(((array([x2, y2]) + t * distance * correction_factor/2)).astype(int))

        cv2.circle(labeled_arr, loc, 10, 255, -1)
        cv2.circle(labeled_arr, loc2, 10, 255, -1)
        cv2.line(labeled_arr, loc2, loc, (0, 255, 0), 10)


        font = cv2.FONT_HERSHEY_PLAIN
        #cv2.putText(labeled_arr, str(int(length))+"cm", tuple(ini.astype(int)), font, 1.5, (255,0,0), 2)
        #if name is not None:
        #    cv2.putText(labeled_arr, name, tuple((ini-20).astype(int)), font, 1.5, (255,0,0), 2)
        #else:
        #    print("none is detected")
        points = (array([x,y]), array([x2,y2]))
        return length, points

    def _calcNormalDistance(self, 
                            points, 
                            arr, 
                            binary, 
                            ratio, 
                            labeled_arr, 
                            max_length=1000,
                            name=None,
                            n_offset=(1, 1),
                            correction_factor=0.0,
                            limit_boundaries=[],
                            ):
        x_lst = []
        y_lst = []
        x2_lst = []
        y2_lst = []
        length_lst = []
        n_lst = []
        for offset_x in range(-n_offset[1], n_offset[1], 1):
            for offset_y in range(-n_offset[0], n_offset[0], 1):
                ini = array([points[0][0] + offset_x, points[0][1] + offset_y])
                t   = array([points[1][0] + offset_x, points[1][1] + offset_y]) - ini

                #points[1][0] += offset_x
                #points[1][1] += offset_y

                #ini = points[0]
                #t = points[1] - ini 
                t = t * 1.
                t /= linalg.norm(t)
                n = cross(append(t, 0), array([0,0,1]))[:-1]
                
                for i in range(max_length):
                    x,y = (ini + i*n).astype(int)
                    if not binary[y, x]:
                        break


                
                for i in range(max_length):
                    x2,y2 = (ini + i*(-n)).astype(int)
                
                    if not binary[y2, x2]:
                        break
                
                length = linalg.norm(array([x2,y2])-array([x,y])) * ratio
                font = cv2.FONT_HERSHEY_PLAIN
                length_lst   += [length]
                x_lst += [x]
                y_lst += [y]
                x2_lst += [x2]
                y2_lst += [y2]
                n_lst += [n]

        length_arr = array(length_lst)
        
        x_arr = array(x_lst)
        y_arr = array(y_lst)
        x2_arr = array(x2_lst)
        y2_arr = array(y2_lst)
        n_arr = array(n_lst)
        
        x2 = x2_arr[length_arr.argmin()]
        y2 = y2_arr[length_arr.argmin()]
        x = x_arr[length_arr.argmin()]
        y = y_arr[length_arr.argmin()]
        n = n_arr[length_arr.argmin()]


        for limit_boundary in limit_boundaries:
            p1, p2 = array([x,y]), array([x2,y2])
            p3, p4, which_restrict, condition = limit_boundary
            point = self._getIntersection(p1, p2, p3, p4)
            dist1 = linalg.norm(p1 - point)
            dist2 = linalg.norm(p2 - point)
            dist3 = linalg.norm(p2 - p1)
            if which_restrict == 2:
                if dist3 > dist2 and condition(p1,p2,p3,p4):
                    x2,y2 = point
            else:
                if dist3 > dist1 and condition(p1,p2,p3,p4):
                    x,y = point

            #cv2.circle(labeled_arr, tuple(limit_boundary[0]), 7, (0, 0, 255), -1)
            #cv2.circle(labeled_arr, tuple(limit_boundary[1]), 7, (0, 255, 0), -1)

        distance = linalg.norm(array([x2,y2]) - array([x,y])) 
        length = distance * ratio * (1 - correction_factor)
                
        loc  = tuple(((array([x, y]) - n * distance * correction_factor/2) ).astype(int))
        loc2 = tuple(((array([x2, y2]) + n * distance * correction_factor/2) ).astype(int))

        cv2.circle(labeled_arr, loc, 10, 255, -1)
        cv2.circle(labeled_arr, loc2, 10, 255, -1)
        cv2.line(labeled_arr, 
                 loc, 
                 loc2, 
                 (0, 255, 0), 10)
                
        points = (array([x,y]), array([x2,y2]))
        return length, points



        
    def estimateSleeve(self, result_dic, point_dic):
        # 袖の長さ
        left_wrist_p1, left_wrist_p2   = point_dic['wrist_left_frontal_width']
        right_wrist_p1, right_wrist_p2 = point_dic['wrist_right_frontal_width']
        shoulder_p1, shoulder_p2       = point_dic['shoulder_width']

        dist1 = linalg.norm(shoulder_p1 - left_wrist_p2)
        dist2 = linalg.norm(shoulder_p2 - right_wrist_p1)

        t1 = (left_wrist_p2  - shoulder_p1) / dist1
        t2 = (right_wrist_p1 - shoulder_p2) / dist2

        dist1 += 2.2 / self.frontal_ratio
        dist2 += 2.2 / self.frontal_ratio

        cv2.line(self.frontal_labeled_arr, tuple(shoulder_p1), tuple((shoulder_p1 + t1 * dist1).astype(int)), (0, 0, 255), 10)
        cv2.line(self.frontal_labeled_arr, tuple(shoulder_p2), tuple((shoulder_p2 + t2 * dist2).astype(int)), (0, 0, 255), 10)

        result_dic['left_sleeve_length']   = left  = linalg.norm(shoulder_p1 - left_wrist_p2) * self.frontal_ratio
        result_dic['right_sleeve_length']  = right = linalg.norm(shoulder_p2 - right_wrist_p1) * self.frontal_ratio
        result_dic['sleeve_length'] = (left + right) / 2
        result_dic['lucas_sleeve_length'] = self.lucas[b'long_sleeve']

        point_dic['left_sleeve']  = (shoulder_p1, left_wrist_p2)
        point_dic['right_sleeve'] = (shoulder_p2, right_wrist_p1)

        return result_dic, point_dic

    def _getIntersection(self, p1, p2, p3, p4):
        a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - a * p1[0] 
    
        c = (p4[1] - p3[1]) / (p4[0] - p3[0])
        d = p3[1] - c * p3[0] 
    
        #print("a={:.2f}, b={:.2f}, c={:.2f}, d={:.2f}".format(a, b, c, d))

        #input(array([(d-b)/(a-c), (a*d-b*c)/(a-c)]).astype(int))
        return array([(d-b)/(a-c), (a*d-b*c)/(a-c)]).astype(int)

    def _calcCrotchPoint(self, contour):
        y = contour[:, 0, 1]
        crit = y.max() - (y.max() - y.min()) * 0.05
        c = crit < y
        contour = contour[c.argmax():c.size - c[::-1].argmax()]
        #for i, p in enumerate(contour):
        #    cv2.circle(self.frontal_resized_outlined_arr, tuple(p[0]), 4, (0, i, i), -1)

        y = contour[:, 0, 1]
        return contour[y.argmin()][0]


    def estimateBodyLength(self, result_dic, point_dic):
        p1 = point_dic['neck_frontal_width'][0]
        t = array([0,1.0])
        p2 = p1 + t * 10
        p3, p4 = point_dic['hem_frontal_width']
        p5 = (p3 + p4) / 2
        p5[0] = p1[0]
        #p5 = self._getIntersection(p1, p2, p3, p4)

        dist = linalg.norm(p5 - p1) + 10 / self.frontal_ratio
        print(p5)
        print(tuple(p1.astype(int)), tuple((p1 + t * dist).astype(int)))
        cv2.line(self.frontal_labeled_arr, tuple(p1.astype(int)), tuple((p1 + t * dist).astype(int)), (0, 0, 255), 10)

        result_dic['body_length'] = dist * self.frontal_ratio
        result_dic['lucas_body_length'] = self.lucas[b'shirt_sleeve']

        return result_dic, point_dic


    def estimateTrouserLength(self, result_dic, point_dic):
        # 袖の長さ
        hem_p1,  hem_p2           = point_dic['hem_frontal_width']
        left_ankle_p1, left_ankle_p2  = point_dic['ankle_left_frontal_width']
        right_ankle_p1, right_ankle_p2  = point_dic['ankle_right_frontal_width']
        left_calf_p1, left_calf_p2  = point_dic['left_calf_width']
        right_calf_p1, right_calf_p2  = point_dic['right_calf_width']

        hem_p1[1] -= hem_p1[1] * 0.05
        hem_p2[1] -= hem_p2[1] * 0.05

        cv2.line(self.frontal_labeled_arr, tuple(hem_p1), tuple(left_ankle_p2), (0, 0, 255), 10)
        cv2.line(self.frontal_labeled_arr, tuple(hem_p2), tuple(right_ankle_p1), (0, 0, 255), 10)


        #crotch_point = self._getIntersection(left_calf_p2, left_ankle_p1, right_calf_p1, right_ankle_p2)
        crotch_point = self._calcCrotchPoint(self.frontal_contour)
        print(crotch_point)

        cv2.line(self.frontal_labeled_arr, tuple(crotch_point), tuple(left_ankle_p1), (0, 0, 255), 10)
        cv2.line(self.frontal_labeled_arr, tuple(crotch_point), tuple(right_ankle_p2), (0, 0, 255), 10)

        result_dic['left_trouser_length']  = left = linalg.norm(hem_p1 - left_ankle_p2) * self.frontal_ratio
        result_dic['right_trouser_length'] = right = linalg.norm(hem_p2 - right_ankle_p1) * self.frontal_ratio
        result_dic['trouser_length']       = (left + right) / 2
        result_dic['left_trouser_inside_length']  = left = linalg.norm(crotch_point  - left_ankle_p1) * self.frontal_ratio
        result_dic['right_trouser_inside_length'] = right = linalg.norm(crotch_point - right_ankle_p2) * self.frontal_ratio
        result_dic['trouser_inside_length']       = (left + right) / 2
        #result_dic['lucas_sleeve_length'] = self.lucas[b'shirt_sleeve']

        return result_dic, point_dic

    def estimateNeck(self, result_dic, point_dic):
        #  首
        result_dic['neck_frontal_width'], point_dic['neck_frontal_width']\
                = neck_frontal_width, _\
                = self._calcNormalDistance(self.frontal_p_dic['neck_width'], 
                                            self.frontal_raw_trimed_extracted_arr, 
                                            self.frontal_binary, 
                                            self.frontal_ratio, 
                                            self.frontal_labeled_arr,
                                            name='neck_frontal_width',
                                            correction_factor=self._getCorrectValue(),
                                            n_offset=(80, 1)
                                            )
        result_dic['neck_side_width'], point_dic['neck_side_width']\
                = neck_side_width, _\
                = self._calcNormalDistance(self.side_p_dic['neck_width'],
                                           self.side_raw_trimed_extracted_arr,
                                           self.side_binary, 
                                           self.side_ratio, 
                                           self.side_labeled_arr,
                                           name='neck_side_width',
                                           correction_factor=self._getCorrectValue(),
                                           n_offset=(80, 1)
                                           )

        result_dic['neck_circumference']\
                = self._calcEllipseLength(
                        neck_frontal_width / 2, 
                        neck_side_width / 2)

        result_dic['lucas_neck_circumference'] = self.lucas[b'neck']

        return result_dic, point_dic


    def estimateShoulderWidth(self, result_dic, point_dic):
        result_dic['shoulder_width'], point_dic['shoulder_width'] = self._calcNormalDistance(
                self.frontal_p_dic['shoulder'], 
                self.frontal_raw_trimed_extracted_arr, 
                self.frontal_binary, 
                self.frontal_ratio, 
                self.frontal_labeled_arr,
                n_offset=(1, 1),
                name='shoulder_width'
                )

        result_dic['lucas_shoulder_width'] = self.lucas[b'shoulder']
        return result_dic, point_dic

    def estimateWaist(self, result_dic, point_dic):

        # 胸回り 
        result_dic['waist_frontal_width'], point_dic['waist_frontal_width']\
                = waist_frontal_width, _\
                = self._calcNormalDistance(self.frontal_p_dic['waist'], 
                                            self.frontal_raw_trimed_extracted_arr, 
                                            self.frontal_binary, 
                                            self.frontal_ratio, 
                                            self.frontal_labeled_arr,
                                            correction_factor=self._getCorrectValue(),
                                            name='waist_frontal_width'
                                            )
        result_dic['waist_side_width'], point_dic['waist_side_width']\
                = waist_side_width, _\
                = self._calcNormalDistance(self.side_p_dic['waist'],
                                           self.side_raw_trimed_extracted_arr,
                                           self.side_binary, 
                                           self.side_ratio, 
                                           self.side_labeled_arr,
                                           name='waist_side_width',
                                           n_offset=(1, 1),
                                           correction_factor=self._getCorrectValue(),
                                           )

        result_dic['waist_circumference']\
                = self._calcEllipseLength(
                        waist_frontal_width / 2, 
                        waist_side_width / 2)

        result_dic['lucas_waist_circumference'] = self.lucas[b'waist']

        return result_dic, point_dic

    def estimateChestWidth(self, result_dic, point_dic):

        # 胸回り 
        p3, p4 = point_dic['shoulder_width'][0], point_dic['waist_frontal_width'][1]
        p5, p6 = point_dic['shoulder_width'][1], point_dic['waist_frontal_width'][0]
        print(p3,p4)
        cv2.line(self.frontal_labeled_arr, tuple(p3), tuple(p4), (0, 0, 255), 2)
        result_dic['chest_frontal_width'], point_dic['chest_frontal_width']\
                = chest_frontal_width, _\
                = self._calcNormalDistance(self.frontal_p_dic['chest'], 
                                            self.frontal_raw_trimed_extracted_arr, 
                                            self.frontal_binary, 
                                            self.frontal_ratio, 
                                            self.frontal_labeled_arr,
                                            name='chest_frontal_width',
                                            n_offset=(10, 1),
                                            correction_factor=self._getCorrectValue(),
                                            limit_boundaries=[(p3, p4, 2, TrueCondition()), (p5, p6, 1, TrueCondition())])
        result_dic['chest_side_width'], point_dic['chest_side_width']\
                = chest_side_width, _\
                = self._calcNormalDistance(self.side_p_dic['chest'],
                                           self.side_raw_trimed_extracted_arr,
                                           self.side_binary, 
                                           self.side_ratio, 
                                           self.side_labeled_arr,
                                           correction_factor=self._getCorrectValue(),
                                           name='chest_side_width'
                                           )

        result_dic['chest_circumference']\
                = self._calcEllipseLength(
                        chest_frontal_width / 2, 
                        chest_side_width / 2)

        result_dic['lucas_chest_circumference'] = self.lucas[b'chest']

        return result_dic, point_dic


    #def estimateGirth(self, result_dic):
    #    # 胴回り
    #    result_dic['girth_frontal_width']\
    #        = girth_frontal_width\
    #        = self._calcTangentDistance(self.frontal_p_dic['girth'], 
    #                                    self.frontal_raw_trimed_extracted_arr, 
    #                                    self.frontal_binary, 
    #                                    self.frontal_ratio)

    #    result_dic['girth_side_width']\
    #        = girth_side_width\
    #        = self._calcNormalDistance(self.side_p_dic['girth'],  
    #                                   self.side_raw_trimed_extracted_arr, 
    #                                   self.side_binary, 
    #                                   self.side_ratio)

    #    result_dic['girth_circumference']\
    #        = self._calcEllipseLength(girth_frontal_width / 2, 
    #                                  girth_side_width / 2)

    #    return result_dic

    def estimateForeArm(self, result_dic, point_dic):
        #裾周り
        result_dic['left_fore_arm_width'], point_dic['left_fore_arm_width']\
            = left_fore_arm_width, _\
            = self._calcNormalDistance(self.frontal_p_dic['left_fore_arm'], 
                                        self.frontal_raw_trimed_extracted_arr, 
                                        self.frontal_binary, 
                                        self.frontal_ratio, 
                                        self.frontal_labeled_arr,
                                        correction_factor=self._getCorrectValue(),
                                        name='left_fore_arm_width')

        result_dic['right_fore_arm_width'], point_dic['right_fore_arm_width']\
            = right_fore_arm_width, _\
            = self._calcNormalDistance(self.frontal_p_dic['right_fore_arm'], 
                                        self.frontal_raw_trimed_extracted_arr, 
                                        self.frontal_binary, 
                                        self.frontal_ratio, 
                                        self.frontal_labeled_arr,
                                        correction_factor=self._getCorrectValue(),
                                        name='right_fore_arm_width')



        result_dic['fore_arm_circumference']\
            = self._calcCircleLength((left_fore_arm_width + right_fore_arm_width) / 4)

        result_dic['fore_arm_length'] = self._calcLength(self.frontal_points, (6,7), (3, 4))


        return result_dic, point_dic

    def _calcLength(self, df, left_indices, right_indices):
        print(self.frontal_ratio)
        p1, p2 = df.ix[left_indices[0]], df.ix[left_indices[1]]
        left = linalg.norm(p1 - p2)
        p1, p2 = df.ix[right_indices[0]], df.ix[right_indices[1]]
        right = linalg.norm(p1 - p2)
        return (left + right) / 2 * self.frontal_ratio * self.frontal_resize_ratio

    def estimateBicep(self, result_dic, point_dic):
        #上腕
        p1, p2 = point_dic['left_sleeve']
        n = (p1-p2) / linalg.norm(p1-p2)

        p3 =  point_dic['left_fore_arm_width'][1]
        p4 = p3 + n
        #cv2.line(self.frontal_labeled_arr, tuple(p3.astype(int)), tuple(p4.astype(int)),  (0, 0, 255), 2)
        result_dic['left_bicep_width'], point_dic['left_bicep_width']\
            = left_bicep_width, _\
            = self._calcNormalDistance(self.frontal_p_dic['left_bicep'], 
                                        self.frontal_raw_trimed_extracted_arr, 
                                        self.frontal_binary, 
                                        self.frontal_ratio, 
                                        self.frontal_labeled_arr,
                                        correction_factor=self._getCorrectValue(),
                                        name='left_bicep_width',
                                        limit_boundaries=[(p3, p4, 2, AngleBoundaryCondition(*self.bicep_critical_value, (0, 1)))])

        p1, p2 = point_dic['right_sleeve']
        n = (p1-p2) / linalg.norm(p1-p2)

        p3 =  point_dic['right_fore_arm_width'][0]
        p4 = p3 + n
        #cv2.line(self.frontal_labeled_arr, tuple(p3.astype(int)), tuple(p4.astype(int)),  (0, 0, 255), 2)
        #cv2.line(self.frontal_labeled_arr, tuple(p3.astype(int)), tuple(p4.astype(int)),  (0, 255, 0), 2)
        #p3, p4 = point_dic['wrist_right_frontal_width'][1], point_dic['right_fore_arm_width'][0]
        result_dic['right_bicep_width'], point_dic['right_bicep_width']\
            = right_bicep_width, _\
            = self._calcNormalDistance(self.frontal_p_dic['right_bicep'], 
                                        self.frontal_raw_trimed_extracted_arr, 
                                        self.frontal_binary, 
                                        self.frontal_ratio, 
                                        self.frontal_labeled_arr,
                                        correction_factor=self._getCorrectValue(),
                                        name='right_bicep_width',
                                        limit_boundaries=[(p3, p4, 1, AngleBoundaryCondition(*self.bicep_critical_value, (0, 1)))]
                                        )


        result_dic['bicep_circumference']\
            = self._calcCircleLength((left_bicep_width + right_bicep_width) / 4)

#        result_dic['bicep_length'] = self._calcLength(\
#                                            self.frontal_p_dic['left_bicep'],
#                                            self.frontal_p_dic['right_bicep'])
#
        result_dic['bicep_length'] = self._calcLength(self.frontal_points, (6,7), (3, 4))

        return result_dic, point_dic

    def estimateThigh(self, result_dic, point_dic):
        #上腕
        result_dic['left_thigh_width'], point_dic['left_thigh_width'], \
            = left_thigh_width, _\
            = self._calcNormalDistance(self.frontal_p_dic['left_thigh'], 
                                        self.frontal_raw_trimed_extracted_arr, 
                                        self.frontal_binary, 
                                        self.frontal_ratio, 
                                        self.frontal_labeled_arr,
                                        correction_factor=self._getCorrectValue(),
                                        name='left_thigh_width')

        result_dic['right_thigh_width'], point_dic['right_thigh_width']\
            = right_thigh_width, _\
            = self._calcNormalDistance(self.frontal_p_dic['right_thigh'], 
                                        self.frontal_raw_trimed_extracted_arr, 
                                        self.frontal_binary, 
                                        self.frontal_ratio, 
                                        self.frontal_labeled_arr,
                                        correction_factor=self._getCorrectValue(),
                                        name='right_thigh_width')



        result_dic['thigh_circumference']\
            = self._calcCircleLength((left_thigh_width + right_thigh_width) / 4)


        result_dic['thigh_length'] = self._calcLength(self.frontal_points, (11,12), (8, 9))
        #result_dic['thigh_length'] = self._calcLength(\
        #                                    self.frontal_p_dic['left_thigh'],
        #                                    self.frontal_p_dic['right_thigh'])

        return result_dic, point_dic

    def estimateCalf(self, result_dic, point_dic):
        #上腕
        result_dic['left_calf_width'], point_dic['left_calf_width']\
            = left_calf_width, _\
            = self._calcNormalDistance(self.frontal_p_dic['left_calf'], 
                                        self.frontal_raw_trimed_extracted_arr, 
                                        self.frontal_binary, 
                                        self.frontal_ratio, 
                                        self.frontal_labeled_arr,
                                        correction_factor=self._getCorrectValue(),
                                        name='left_calf_width')

        result_dic['right_calf_width'], point_dic['right_calf_width']\
            = right_calf_width, _\
            = self._calcNormalDistance(self.frontal_p_dic['right_calf'], 
                                        self.frontal_raw_trimed_extracted_arr, 
                                        self.frontal_binary, 
                                        self.frontal_ratio, 
                                        self.frontal_labeled_arr,
                                        correction_factor=self._getCorrectValue(),
                                        name='right_calf_width')



        result_dic['calf_circumference']\
            = self._calcCircleLength((left_calf_width + right_calf_width) / 4)

        result_dic['calf_length'] = self._calcLength(self.frontal_points, (12,13), (9, 10))
        #result_dic['calf_length'] = self._calcLength(\
        #                                    self.frontal_p_dic['left_calf'],
        #                                    self.frontal_p_dic['right_calf'])

        return result_dic, point_dic

    def estimateAnkle(self, result_dic, point_dic):
        #裾周り
        result_dic['ankle_left_frontal_width'], point_dic['ankle_left_frontal_width']\
            = ankle_left_frontal_width, _\
            = self._calcNormalDistance(self.frontal_p_dic['left_ankle'], 
                                        self.frontal_raw_trimed_extracted_arr, 
                                        self.frontal_binary, 
                                        self.frontal_ratio, 
                                        self.frontal_labeled_arr,
                                        correction_factor=self._getCorrectValue(),
                                        name='ankle_left_frontal_width')

        result_dic['ankle_right_frontal_width'], point_dic['ankle_right_frontal_width']\
            = ankle_right_frontal_width, _\
            = self._calcNormalDistance(self.frontal_p_dic['right_ankle'], 
                                        self.frontal_raw_trimed_extracted_arr, 
                                        self.frontal_binary, 
                                        self.frontal_ratio, 
                                        self.frontal_labeled_arr,
                                        correction_factor=self._getCorrectValue(),
                                        name='ankle_right_frontal_width')


        result_dic['ankle_circumference']\
            = self._calcCircleLength((ankle_left_frontal_width + ankle_right_frontal_width) / 4)

        return result_dic, point_dic

    def estimateWrist(self, result_dic, point_dic):
        #裾周り
        result_dic['wrist_left_frontal_width'], point_dic['wrist_left_frontal_width']\
            = wrist_left_frontal_width, _\
            = self._calcNormalDistance(self.frontal_p_dic['left_wrist'], 
                                        self.frontal_raw_trimed_extracted_arr, 
                                        self.frontal_binary, 
                                        self.frontal_ratio, 
                                        self.frontal_labeled_arr,
                                        correction_factor=self._getCorrectValue(),
                                        name='wrist_left_frontal_width')

        result_dic['wrist_right_frontal_width'], point_dic['wrist_right_frontal_width']\
            = wrist_right_frontal_width, _\
            = self._calcNormalDistance(self.frontal_p_dic['right_wrist'], 
                                        self.frontal_raw_trimed_extracted_arr, 
                                        self.frontal_binary, 
                                        self.frontal_ratio, 
                                        self.frontal_labeled_arr,
                                        correction_factor=self._getCorrectValue(),
                                        name='wrist_right_frontal_width')


        result_dic['wrist_circumference']\
            = self._calcCircleLength((wrist_left_frontal_width + wrist_right_frontal_width) / 4)

        return result_dic, point_dic

    def estimateHem(self, result_dic, point_dic):
        #裾周り
        result_dic['hem_frontal_width'], point_dic['hem_frontal_width']\
            = hem_frontal_width, _\
            = self._calcTangentDistance(self.frontal_p_dic['hem'], 
                                        self.frontal_raw_trimed_extracted_arr, 
                                        self.frontal_binary, 
                                        self.frontal_ratio, 
                                        self.frontal_labeled_arr,
                                        correction_factor=self._getCorrectValue(),
                                        name='hem_frontal_width')

        result_dic['hem_side_width'], point_dic['hem_side_width']\
            = hem_side_width, _\
            = self._calcNormalDistance(self.side_p_dic['hem'],  
                                       self.side_raw_trimed_extracted_arr, 
                                       self.side_binary, 
                                       self.side_ratio, 
                                       self.side_labeled_arr,
                                       correction_factor=0.1,
                                       name='hem_side_width')

        result_dic['hem_circumference']\
            = self._calcEllipseLength(hem_frontal_width / 2, 
                                      hem_side_width / 2)

        result_dic['lucas_hem_circumference'] = self.lucas[b'hip']

        return result_dic, point_dic

    def _calcEllipseLength(self, a, b):
        return pi * (a + b) * (1 + (3 * ((a - b) / (a + b)) ** 2) / (10 + sqrt(4 - 3 * (a - b) / (a + b)) ** 2))

    def _calcCircleLength(self, a):
        return 2 * pi * a

    def _extractBackgroundOfHuman(self, raw_image, resize_ratio, transform='', gpu_id=0, divide_size=(1,1), dump_path='tmp', image_path=None):
        h,w,c = raw_image.shape
        # coding: utf-8
        import aimaker.predictor.segmentation_predictor as sp
        #a = sp.SegmentationPredictor("/home/yu/Dropbox/manechin", "resize256x305_toNumpy", gpu_ids='0',divide_size=(2,1))
        if len(transform) == 0:
            transform = "toNumpy"
        else:
            transform += "_toNumpy"

        a = sp.SegmentationPredictor(image_path, transform, gpu_ids=str(gpu_id),divide_size=divide_size)
        #print(image.astype(uint8))
        #print(image.astype(uint8).shape)
       # print("kokodesu {}".format(raw_image.shape))
        result_raw_image = a._dividedPredict(I.fromarray(raw_image.astype(uint8)), resize_ratio=(1/resize_ratio, 1/resize_ratio))
        result_resized_image = cv2.resize(result_raw_image, (int(w / resize_ratio), int(h / resize_ratio)), interpolation=cv2.INTER_CUBIC)
        return result_resized_image, result_raw_image



    def _estimatePose(self, raw_image, draw_image, resize_ratio, dump_path=None, gpu_id=0, weight_name = './model/pose_model.pth'):
        import os
        import re
        import sys
        import cv2
        import math
        import time
        import scipy
        import argparse
        import matplotlib
        from torch import np
        import pylab as plt
        from joblib import Parallel, delayed
        import util
        import torch
        import torch as T
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.autograd import Variable
        from collections import OrderedDict
        from config_reader import config_reader
        from scipy.ndimage.filters import gaussian_filter
        
        torch.set_num_threads(torch.get_num_threads())
        
        blocks = {}
        
        # find connection in the specified sequence, center 29 is in the position 15
        limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
                   [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
                   [1,16], [16,18], [3,17], [6,18]]
                   
        # the middle joints heatmap correpondence
        mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
                  [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
                  [55,56], [37,38], [45,46]]
                  
        # visualize
        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
                  
                     
        block0  = [{'conv1_1':[3,64,3,1,1]},{'conv1_2':[64,64,3,1,1]},{'pool1_stage1':[2,2,0]},{'conv2_1':[64,128,3,1,1]},{'conv2_2':[128,128,3,1,1]},{'pool2_stage1':[2,2,0]},{'conv3_1':[128,256,3,1,1]},{'conv3_2':[256,256,3,1,1]},{'conv3_3':[256,256,3,1,1]},{'conv3_4':[256,256,3,1,1]},{'pool3_stage1':[2,2,0]},{'conv4_1':[256,512,3,1,1]},{'conv4_2':[512,512,3,1,1]},{'conv4_3_CPM':[512,256,3,1,1]},{'conv4_4_CPM':[256,128,3,1,1]}]
        
        blocks['block1_1']  = [{'conv5_1_CPM_L1':[128,128,3,1,1]},{'conv5_2_CPM_L1':[128,128,3,1,1]},{'conv5_3_CPM_L1':[128,128,3,1,1]},{'conv5_4_CPM_L1':[128,512,1,1,0]},{'conv5_5_CPM_L1':[512,38,1,1,0]}]
        
        blocks['block1_2']  = [{'conv5_1_CPM_L2':[128,128,3,1,1]},{'conv5_2_CPM_L2':[128,128,3,1,1]},{'conv5_3_CPM_L2':[128,128,3,1,1]},{'conv5_4_CPM_L2':[128,512,1,1,0]},{'conv5_5_CPM_L2':[512,19,1,1,0]}]
        
        for i in range(2,7):
            blocks['block%d_1'%i]  = [{'Mconv1_stage%d_L1'%i:[185,128,7,1,3]},{'Mconv2_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv3_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv4_stage%d_L1'%i:[128,128,7,1,3]},
        {'Mconv5_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv6_stage%d_L1'%i:[128,128,1,1,0]},{'Mconv7_stage%d_L1'%i:[128,38,1,1,0]}]
            blocks['block%d_2'%i]  = [{'Mconv1_stage%d_L2'%i:[185,128,7,1,3]},{'Mconv2_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv3_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv4_stage%d_L2'%i:[128,128,7,1,3]},
        {'Mconv5_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv6_stage%d_L2'%i:[128,128,1,1,0]},{'Mconv7_stage%d_L2'%i:[128,19,1,1,0]}]
        
        def make_layers(cfg_dict):
            layers = []
            for i in range(len(cfg_dict)-1):
                one_ = cfg_dict[i]
                for k,v in one_.items():      
                    if 'pool' in k:
                        layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
                    else:
                        conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
                        layers += [conv2d, nn.ReLU(inplace=True)]
            one_ = list(cfg_dict[-1].keys())
            k = one_[0]
            v = cfg_dict[-1][k]
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
            layers += [conv2d]
            return nn.Sequential(*layers)
            
        layers = []
        for i in range(len(block0)):
            one_ = block0[i]
            for k,v in one_.items():      
            #for k,v in one_.iteritems():      
                if 'pool' in k:
                    layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
                else:
                    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
                    layers += [conv2d, nn.ReLU(inplace=True)]  
               
        models = {}           
        models['block0']=nn.Sequential(*layers)        
        
        for k,v in blocks.items():
            models[k] = make_layers(v)
                        
        class pose_model(nn.Module):
            def __init__(self,model_dict,transform_input=False):
                super(pose_model, self).__init__()
                self.model0   = model_dict['block0']
                self.model1_1 = model_dict['block1_1']        
                self.model2_1 = model_dict['block2_1']  
                self.model3_1 = model_dict['block3_1']  
                self.model4_1 = model_dict['block4_1']  
                self.model5_1 = model_dict['block5_1']  
                self.model6_1 = model_dict['block6_1']  
                
                self.model1_2 = model_dict['block1_2']        
                self.model2_2 = model_dict['block2_2']  
                self.model3_2 = model_dict['block3_2']  
                self.model4_2 = model_dict['block4_2']  
                self.model5_2 = model_dict['block5_2']  
                self.model6_2 = model_dict['block6_2']
                
            def forward(self, x):    
                out1 = self.model0(x)
                
                out1_1 = self.model1_1(out1)
                out1_2 = self.model1_2(out1)
                out2  = torch.cat([out1_1,out1_2,out1],1)
                
                out2_1 = self.model2_1(out2)
                out2_2 = self.model2_2(out2)
                out3   = torch.cat([out2_1,out2_2,out1],1)
                
                out3_1 = self.model3_1(out3)
                out3_2 = self.model3_2(out3)
                out4   = torch.cat([out3_1,out3_2,out1],1)
        
                out4_1 = self.model4_1(out4)
                out4_2 = self.model4_2(out4)
                out5   = torch.cat([out4_1,out4_2,out1],1)  
                
                out5_1 = self.model5_1(out5)
                out5_2 = self.model5_2(out5)
                out6   = torch.cat([out5_1,out5_2,out1],1)         
                      
                out6_1 = self.model6_1(out6)
                out6_2 = self.model6_2(out6)
                
                return out6_1,out6_2        
        
        
        model = pose_model(models)     
        model.load_state_dict(torch.load(weight_name))
        model.cuda(gpu_id)
        model.float()
        model.eval()
        
        param_, model_ = config_reader()
        
        tic = time.time()

        #test_image = 'frontal_yu_result.jpg'
        #test_image = 'frontal_yu_result_1028.jpg'
        #test_image = 'frontal_yu_result_1028.png'
        #test_image = './sample_image/front.jpg'
        #test_image = 'a.jpg'
        oriImg = raw_image.copy()
        imageToTest = Variable(T.transpose(T.transpose(T.unsqueeze(torch.from_numpy(oriImg).float(),0),2,3),1,2),volatile=True).cuda(gpu_id)
        
        multiplier = [x * model_['boxsize'] / oriImg.shape[0] for x in param_['scale_search']]
        
        heatmap_avg = torch.zeros((len(multiplier),19,oriImg.shape[0], oriImg.shape[1])).cuda(gpu_id)
        paf_avg     = torch.zeros((len(multiplier),38,oriImg.shape[0], oriImg.shape[1])).cuda(gpu_id)
        #print heatmap_avg.size()
        
        toc =time.time()
        #print ('time is %.5f'%(toc-tic) )
        tic = time.time()
        for m in range(len(multiplier)):
            scale = multiplier[m]
            h = int(oriImg.shape[0]*scale)
            w = int(oriImg.shape[1]*scale)
            pad_h = 0 if (h%model_['stride']==0) else model_['stride'] - (h % model_['stride']) 
            pad_w = 0 if (w%model_['stride']==0) else model_['stride'] - (w % model_['stride'])
            new_h = h+pad_h
            new_w = w+pad_w
        
            imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_['stride'], model_['padValue'])
            imageToTest_padded = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5
            
            feed = Variable(T.from_numpy(imageToTest_padded)).cuda(gpu_id)      
            output1,output2 = model(feed)
            #print (output1.size())
            #print (output2.size())
            heatmap = nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])).cuda(gpu_id)(output2)
            
            paf = nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])).cuda(gpu_id)(output1)       
        
            heatmap_avg[m] = heatmap[0].data
            paf_avg[m] = paf[0].data  
            
            
        toc = time.time()
        #print( 'time is %.5f'%(toc-tic) )
        tic = time.time()
            
        heatmap_avg = T.transpose(T.transpose(T.squeeze(T.mean(heatmap_avg, 0)),0,1),1,2).cuda(gpu_id) 
        paf_avg     = T.transpose(T.transpose(T.squeeze(T.mean(paf_avg, 0)),0,1),1,2).cuda(gpu_id) 
        heatmap_avg=heatmap_avg.cpu().numpy()
        paf_avg    = paf_avg.cpu().numpy()
        toc =time.time()
        #print( 'time is %.5f'%(toc-tic) )
        tic = time.time()
        
        all_peaks = []
        peak_counter = 0
        
        #maps = 
        for part in range(18):
            map_ori = heatmap_avg[:,:,part]
            map = gaussian_filter(map_ori, sigma=3)
            
            map_left = np.zeros(map.shape)
            map_left[1:,:] = map[:-1,:]
            map_right = np.zeros(map.shape)
            map_right[:-1,:] = map[1:,:]
            map_up = np.zeros(map.shape)
            map_up[:,1:] = map[:,:-1]
            map_down = np.zeros(map.shape)
            map_down[:,:-1] = map[:,1:]
            
            peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param_['thre1']))
        #    peaks_binary = T.eq(
        #    peaks = zip(T.nonzero(peaks_binary)[0],T.nonzero(peaks_binary)[0])
            
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
            
            peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
            id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
        
            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)
            
            
            
            
            
        connection_all = []
        special_k = []
        mid_num = 10
        
        for k in range(len(mapIdx)):
            score_mid = paf_avg[:,:,[x-19 for x in mapIdx[k]]]
            candA = all_peaks[limbSeq[k][0]-1]
            candB = all_peaks[limbSeq[k][1]-1]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = limbSeq[k]
            if(nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                        vec = np.divide(vec, norm)
                        
                        startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                       np.linspace(candA[i][1], candB[j][1], num=mid_num)))
                        
                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                          for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                          for I in range(len(startend))])
        
                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)
                        criterion1 = len(np.nonzero(score_midpts > param_['thre2'])[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])
        
                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0,5))
                for c in range(len(connection_candidate)):
                    i,j,s = connection_candidate[c][0:3]
                    if(i not in connection[:,3] and j not in connection[:,4]):
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if(len(connection) >= min(nA, nB)):
                            break
        
                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])
        
        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])
        
        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:,0]
                partBs = connection_all[k][:,1]
                indexA, indexB = np.array(limbSeq[k]) - 1
        
                for i in range(len(connection_all[k])): #= 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)): #1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1
                    
                    if found == 1:
                        j = subset_idx[0]
                        if(subset[j][indexB] != partBs[i]):
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2: # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        #print ("found = 2")
                        membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0: #merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else: # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
        
                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])
        
        # delete some rows of subset which has few parts occur
        deleteIdx = [];
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)
        
        canvas = draw_image.copy()
        #o = open("frontal_kenji_axis.txt", "w")
        o = open("side_man_axis.txt", "w")
        lst = []
        for i in range(18):
            for j in range(len(all_peaks[i])):
                font = cv2.FONT_HERSHEY_SIMPLEX
                tmp = all_peaks[i][j][0:2]
                loc = (int(tmp[0] * resize_ratio), int(tmp[1] * resize_ratio))
                #print(loc)
                cv2.putText(canvas, "{}".format(i), loc, font, 4, colors[i], 3)
                loc = all_peaks[i][j][0:2]
                tmp = []
                tmp += [int(loc[0] * resize_ratio)]
                tmp += [int(loc[1] * resize_ratio)]
                loc = tmp
                cv2.circle(canvas, tuple(loc), 10, colors[i], thickness=-1)
                o.write("{},{},{}\n".format(i,*all_peaks[i][j][0:2]))
                lst += [[i,*all_peaks[i][j][0:2]]]
                
        param_arr = array(lst) 
        
        stickwidth = 20
        
        for i in range(17):
            for n in range(len(subset)):
                index = subset[n][np.array(limbSeq[i])-1]
                if -1 in index:
                    continue
                cur_canvas = canvas.copy()
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]
                mX = np.mean(X) * resize_ratio
                mY = np.mean(Y) * resize_ratio
                length = resize_ratio * ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        
        #Parallel(n_jobs=1)(delayed(handle_one)(i) for i in range(18))
        
        toc =time.time()
        #print ('time is %.5f'%(toc-tic))
        if dump_path is not None:
            cv2.imwrite(dump_path, canvas)   

        return param_arr, canvas[...,::-1]
        #cv2.imwrite('result.png',canvas)   

