%% Jacard Index
%% Assigning the folder

GrabCut_image_folder = 'C:\SKINLESIONSEGMENTATION\grabcut_hsv_bw';%  Enter name of folder from which you want to upload pictures with full path
GroundT_image_folder=  'C:\SKINLESIONSEGMENTATION\rio';
Segmented_Mask_folder= 'C:\SKINLESIONSEGMENTATION\grabcut_hsv_bw';

%% Calculating the file names and Length of images
% %% Grabcut Directory info
GrabCut_filenames = dir(fullfile(GrabCut_image_folder, '*.png'));  % read all images with specified extention, its jpg in our case
GrabCut_total_images = numel(GrabCut_filenames);

% Ground Truth Directory info
GroundT_filenames = dir(fullfile(GroundT_image_folder, '*.png'));  % read all images with specified extention, its jpg in our case
GroundT_total_images = numel(GroundT_filenames);

% Segmented Mask directory
Segemnted_Mask_filenames = dir(fullfile(Segmented_Mask_folder, '*.png'));  % read all images with specified extention, its jpg in our case
Segmented_Mask_total_images = numel(Segemnted_Mask_filenames);

cell={};
 
 for n = 1:GroundT_total_images                        
  GrabCut_full_name= fullfile(GrabCut_image_folder, GrabCut_filenames(n).name);% it will specify images names with full path and extension
  GrabCut_our_images = imread(GrabCut_full_name);
  
  GroundT_full_name= fullfile(GroundT_image_folder, GroundT_filenames(n).name);% it will specify images names with full path and extension
  GroundT_our_images = imread(GroundT_full_name);
  
  Segmented_full_name= fullfile(Segmented_Mask_folder, Segemnted_Mask_filenames(n).name);% it will specify images names with full path and extension
  Segmented_our_images = imread(Segmented_full_name);
  
  result=eval_metrics(Segmented_our_images,GroundT_our_images);
  
   cell{n,1}=GroundT_filenames(n).name;
   cell{n,2}=result;
 end