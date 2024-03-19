import sys
import os
import requests
import zipfile
import shutil
import json
import cv2
import numpy as np

import src.ich_system_lib as ichlib 

from tensorflow import keras 



#data_dir = '/home/data'
#local_out_dir = 'outputs'
system_detail = "ich : ppc with window ich,posneg_resnet101_ep15_lr5e-6,ich_type_resnet152_ep5_lr2e-5"

ich_posneg_model_pth = r"model/ich_posneg.h5" 
ich_type_model_pth = r"model/ich_type.h5" 
wb_prediction_mode = True


def ich_sys(a,b):
    #print ("hello and that's your sum: " + str(a) +"  "+ str(b))
    #in_fpath = r'test_input\pttest.zip'#r'test_input\5c2545a1-6916-42ca-b2f0-bb682f58c435_data_in.zip'#'test_input\ptid220524.zip'  # DEBUG purpose
    in_dir = a #os.path.join(local_out_dir, 'in')
    out_dir = b #os.path.join(local_out_dir, 'out')
    out_dicom_dir = os.path.join(out_dir,"dicom")
    out_png_dir = os.path.join(out_dir,"png")

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir,exist_ok=True)

    if not os.path.isdir(out_dicom_dir):
        os.makedirs(out_dicom_dir,exist_ok=True)

    if not os.path.isdir(out_png_dir):
        os.makedirs(out_png_dir,exist_ok=True)

    #out_file = os.path.join(local_out_dir, ".zip")


    try :
        # with zipfile.ZipFile(in_fpath, 'r') as zip_ref:
        #     zip_ref.extractall(in_dir)
        ich_type_model = keras.models.load_model(ich_type_model_pth)
        ich_posneg_model = keras.models.load_model(ich_posneg_model_pth)

        ich_type_model = keras.models.load_model(ich_type_model_pth)
        ich_posneg_model = keras.models.load_model(ich_posneg_model_pth)

        dcm_in_dir = os.path.join(in_dir)  #os.path.join(in_dir,"dcm")
        fname = os.listdir(dcm_in_dir)
        # print(fname)
        output_folder = out_dir
        json_output_folder = out_dir
        dcm_output_folder = out_dicom_dir
        pred_output_folder = out_png_dir
        seg_output_folder = out_png_dir

        os.makedirs(output_folder, exist_ok= True)
        os.makedirs(dcm_output_folder, exist_ok= True)
        os.makedirs(pred_output_folder, exist_ok= True)
        os.makedirs(json_output_folder, exist_ok= True)

        dicom_dict = []
        img1_pth_ls = []
        img1_ls = []
        img2_pth_ls = []
        img2_ls = []
        dcm_pth_ls = []
        dcm_save_ls = []
        ds_dcm_save_ls = []
        pred_ich_ls = []
        tag_result_img = []
        str_pred_ls = []
        ich_prob = []
        ov_stt_any_pb = []
        ich_overall_report = "single-slice , Not-Detected"


        for f in range (0,len(fname)) :
            print(str(f) + " / " + str(len(fname)))
            ############## Set file name and path 
            split_filename = fname[f].split(".")
            if len(split_filename) == 2 :
                img_name = split_filename[0]
                ext = "."+ split_filename[1]
            else :
                nfname = len(fname[f]) 
                if fname[f][nfname-4 : nfname] == ".dcm" :
                    img_name = fname[f][:nfname-4]
                    ext = ".dcm"
                    #print(img_name)
                else :
                    img_name = fname[f]
                    ext = ""

            img1_name = img_name + "_ich" +".png"
            img2_name = img_name + str(f) + "_seg" +".png"
            #img_wich_name = "IM" + str(f) + "_wich" +".jpg"

            dcm_pth = os.path.join(dcm_in_dir,img_name+ext) #set input dcm path
            output_dcm_pth = os.path.join(dcm_output_folder,img_name+"_ich.dcm")  #set output dcm path
            output_im1_pth = os.path.join(pred_output_folder,img1_name) 
            output_im2_pth = os.path.join(seg_output_folder,img2_name)
            #output_im_wich_pth = os.path.join(output_folder,img_wich_name)

            

            # ############# dicom preprocess
            hu,ds = ichlib.get_hu(dcm_pth)

            high_th,user_font,ich_any_header_user_font_scale,status_border_textcolour,status_textcolour,no_digit,thickness_border_header,thickness_text_header = ichlib.get_var()

            if wb_prediction_mode == True :
                im1,im2,tag_result,savingdcm,ich_status,status_any_prob,ov_status_ich_type_and_any_pb,str_pred = ichlib.ich_prediction (hu,ich_posneg_model,ich_type_model)

                slice_prediction = {
                "name": img_name ,
                "input_fname" : fname[f],
                "status": ich_status,
                "img1": img1_name,
                "img2": "",#img2_name,
                "data" : str_pred
                }
        

                
                if status_any_prob < 0 :
                    ich_status_int = -1
                elif status_any_prob > high_th :
                    ich_status_int = 1
                else:
                    ich_status_int = 0
                
                
                pred_ich_ls.append(ich_status_int)
                img1_ls.append(im1)
                img1_pth_ls.append(output_im1_pth)
                img2_ls.append(im2)
                img2_pth_ls.append(output_im2_pth)
                dcm_pth_ls.append(output_dcm_pth)
                dcm_save_ls.append(savingdcm)
                ds_dcm_save_ls.append(ds)
                str_pred_ls.append(str_pred)

                dicom_dict.append(slice_prediction)
                tag_result_img.append(tag_result)
                ich_prob.append(status_any_prob*100)
                ov_stt_any_pb.append(ov_status_ich_type_and_any_pb)
            else :
                im1,im2,savingdcm,ich_status,status_any_prob,str_pred = ichlib.ich_prediction_single_slice (hu,ich_posneg_model,ich_type_model)
                if ich_status == "Detected" :
                    ich_overall_report = "single-slice , Detected"
                else :
                    ich_overall_report = ich_overall_report

                slice_prediction = {
                "name": img_name ,
                "input_fname" : fname[f],
                "status": ich_status,
                "img1": img1_name,
                "img2": "",#img2_name,
                "data" : str_pred
                }
                #print("single_slice")
                dicom_dict.append(slice_prediction)
                cv2.imwrite(output_im1_pth,im1)
                ichlib.savedcm(savingdcm,ds,output_dcm_pth,str_pred)

        #################
        
        if wb_prediction_mode == True :
            count_pos = 0
            prev_count_pos = 0
            prev_pos_cnt_index = []
            pos_cnt_index = []
            
            #print(pred_ich_ls)
            for l in range (1,len(pred_ich_ls)-2) :
                if pred_ich_ls[l-1] == 1 and pred_ich_ls[l] == 1 and pred_ich_ls[l+1] == 1 and pred_ich_ls[l+2] == 0 : 
                    if prev_count_pos < count_pos :
                        prev_count_pos = count_pos
                        prev_pos_cnt_index = pos_cnt_index
                    else :
                        prev_count_pos = prev_count_pos
                        prev_pos_cnt_index = prev_pos_cnt_index
                elif pred_ich_ls[l-1] == 1 and pred_ich_ls[l] == 1 and pred_ich_ls[l+1] == 1 :
                    if count_pos == 0 :
                        count_pos = 3
                        pos_cnt_index.append(l-1)
                        pos_cnt_index.append(l)
                        pos_cnt_index.append(l+1)
                    else : #count_pos > prev_count_pos :
                        count_pos = count_pos + 1
                        pos_cnt_index.append(l+1)
                else :
                    count_pos = 0
                    pos_cnt_index = []
        
            total_prev_count_pos = prev_count_pos + 1

            total_count_pos = count_pos + 1
            
            # print(total_prev_count_pos)
            # print(prev_pos_cnt_index)
            # print(total_count_pos)
            #print(pos_cnt_index)

            if len(pos_cnt_index) > 0 or len(prev_pos_cnt_index) > 0:

                if count_pos > prev_count_pos :
                    total_count_pos = count_pos + 1
                    pos_cnt_index.append(int(pos_cnt_index[-1]+1))
                    total_count_pos_index = pos_cnt_index
                elif prev_count_pos > count_pos :
                    total_count_pos = prev_count_pos + 1
                    prev_pos_cnt_index.append(int(prev_pos_cnt_index[-1]+1))
                    total_count_pos_index = prev_pos_cnt_index
                else :
                    total_count_pos = count_pos + 1
                    pos_cnt_index.append(int(pos_cnt_index[-1]+1))
                    total_count_pos_index = pos_cnt_index

                #print(total_count_pos)
                #print(total_count_pos_index)
                #print(total_count_pos%2)
                ich_overview_index = int(np.mean(total_count_pos_index))
                #print(pred_ich_ls[ich_overview_index])
                overview_ich_img2= img2_ls[ich_overview_index]
                overview_ich_img1 = img1_ls[ich_overview_index]
                overview_result_tag = tag_result_img[ich_overview_index]

                ich_overall_report = "whole-brain,detected,overview of whole-brain slice = " + str(ich_overview_index)
            elif len(pred_ich_ls) > 0 :
                max_porb = np.max(ov_stt_any_pb)
                index_max_prob = np.where(ov_stt_any_pb==max_porb)[0]

                if len(index_max_prob) > 1 :
                    index_ich_neg_choose = int(np.median(index_max_prob))
                elif len(index_max_prob) > 0 :
                    index_ich_neg_choose = int(index_max_prob[0])
                else :
                    index_ich_neg_choose = 0

                if max_porb < 50  :
                    colour_ov_not_continue = (0,125,0)
                else :
                    colour_ov_not_continue = (45,75,105)


                # user_font = cv2.FONT_HERSHEY_SIMPLEX
                # ich_any_header_user_font_scale = 0.7
                # status_border_textcolour = (0,0,0)
                # status_textcolour =(255,255,255)
                # no_digit = None


                ich_header = str(round(max_porb,no_digit)) + " %"
                tag_new = tag_result_img[index_ich_neg_choose]

                cv2.rectangle(tag_new, (0, 0), (115, 50), colour_ov_not_continue, cv2.FILLED)
                cv2.putText(img=tag_new, text= "WB-ICH"  , org=(15,20), fontFace=user_font, fontScale=ich_any_header_user_font_scale, color=status_border_textcolour,thickness=thickness_border_header)
                cv2.putText(img=tag_new, text= ich_header , org=(25,45), fontFace=user_font, fontScale=ich_any_header_user_font_scale, color=status_border_textcolour,thickness=thickness_border_header)
                cv2.putText(img=tag_new, text= "WB-ICH"  , org=(15,20), fontFace=user_font, fontScale=ich_any_header_user_font_scale, color=status_textcolour,thickness=thickness_text_header)
                cv2.putText(img=tag_new, text= ich_header , org=(25,45), fontFace=user_font, fontScale=ich_any_header_user_font_scale, color=status_textcolour,thickness=thickness_text_header)
                
                overview_ich_img2= img2_ls[index_ich_neg_choose]
                overview_ich_img1 = img1_ls[index_ich_neg_choose]
                overview_result_tag = tag_new

                ich_overall_report = "whole-brain,detected with conditions, overview of whole-brain slice = " + str(index_ich_neg_choose)
            else :
                ich_overall_report = "whole-brain,undetected : len of input file < 1"

            for s in range (0,len(pred_ich_ls)) :
                bg_result = img1_ls[s]

                alpha_overview = 0.5
                report_bgr = bg_result.copy()

                overview_percent = 25
                new_width = int(overview_ich_img2.shape[1]*overview_percent/100)
                new_height = int(overview_ich_img2.shape[0]*overview_percent/100)

                overview_brain_bgr = cv2.resize(overview_ich_img2, (new_width, new_height))
                overview_result_tag_bgr = overview_result_tag #overview_ich_img1[461:][:]

                x_offset=0
                y_offset=40
                inf_x_offset = 0
                inf_y_offset = 0

                crop_bg  = bg_result[y_offset:y_offset+overview_brain_bgr.shape[0], x_offset:x_offset+overview_brain_bgr.shape[1]]
                report_img_ov = crop_bg.copy()
                img_ov_br = overview_brain_bgr.copy()

                alpha_mask_ov_br = 0.3
                mask_ov_br = img_ov_br.astype(bool)

                report_img_ov[mask_ov_br] = cv2.addWeighted(crop_bg, alpha_mask_ov_br, img_ov_br, 1 - alpha_mask_ov_br, 0)[mask_ov_br]

                report_bgr[y_offset:y_offset+overview_brain_bgr.shape[0], x_offset:x_offset+overview_brain_bgr.shape[1]] = report_img_ov #overview_brain_bgr
                report_bgr[inf_y_offset:inf_y_offset+overview_result_tag_bgr.shape[0], inf_x_offset:inf_x_offset+overview_result_tag_bgr.shape[1]] = overview_result_tag_bgr

                frspp_fs = 0.5 #ich_any_header_user_font_scale #0.7 
                frspp_bd_tn = 5 #thickness_border_header #5 
                frspp_tn =  1 #thickness_text_header #2

                # print(frspp_fs)
                # print(frspp_bd_tn)
                # print(frspp_tn)
                cv2.putText(img=report_bgr, text= "For research purpose only"  , org=(275,450), fontFace=user_font, fontScale=frspp_fs, color=status_border_textcolour,thickness=frspp_bd_tn)
                cv2.putText(img=report_bgr, text= "For research purpose only"  , org=(275,450), fontFace=user_font, fontScale=frspp_fs, color=status_textcolour,thickness=frspp_tn)


                report_rgb = cv2.cvtColor(report_bgr,cv2.COLOR_BGR2RGB)
                cv2.imwrite(img1_pth_ls[s],report_bgr)
                ichlib.savedcm(report_rgb,ds_dcm_save_ls[s],dcm_pth_ls[s],str_pred_ls[s])


        ##################################
        ############ Write Json
        json_pth = os.path.join(json_output_folder,str("ich_pred.json"))
        ctscen_prediction = {
            "process_status" : "complete",
            "error_code" : "",
            "studyQueueID" : "Qid",
            "sys_detial" : system_detail,
            "ich_overall_status" : "",
            "ich_type_status" :  ""  ,
            "ich_report" : ich_overall_report,
            "ich_overall_prob " : "",
            "dicom" : dicom_dict
        }

        json_data=json.dumps(ctscen_prediction)

        with open(json_pth, "w") as outfile:
            outfile.write(json_data)

        #print("Finish!!")

    except Exception as e:
        json_pth = os.path.join(json_output_folder,str("ich_pred.json"))
        ctscen_prediction = {
            "process_status" : "error",
            "error_code" : str(e),
            "studyQueueID" : "Qid",
            "sys_detial" : system_detail,
            "ich_overall_status" : "",
            "ich_type_status" :  ""  ,
            "ich_report" : ich_overall_report,
            "ich_overall_prob " : "",
            "dicom" : ""
        }
        #print(e)
        json_data=json.dumps(ctscen_prediction)

        with open(json_pth, "w") as outfile:
            outfile.write(json_data)

    #shutil.make_archive(out_file.split('.')[0], 'zip', out_dir)
    print("saved output to " + str(out_dir))

    return True


if __name__ == "__main__":
    input_fd = sys.argv[1]
    output_fd = sys.argv[2]
    
    ich_sys(input_fd, output_fd)