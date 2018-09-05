function r=eval_metrics(bw,ref_bw)
% INPUTS
%     bw: segmented image-(logical)
%     ref_bw: ground-truth-(logical)
%     bw_mask: FOV mask-(logical)
% OUTPUTS
%     r=[TPR;FPR;accuracy;precision];
%
TP_image=ref_bw&bw;
TP=sum(TP_image(:));% # of hits (True Positive)
FN_image=ref_bw&~bw;
FN=sum(FN_image(:));% # of misses (False Negative\Type 2 Error)

FP_image=~ref_bw&bw;
FP=sum(FP_image(:));% # of false alarms (False Positive/Type 1 Error)
TN_image=~ref_bw&~bw;
TN=sum(TN_image(:));% # of correct rejections (True Negative)

accuracy=(TP+TN)/(TP+FN+FP+TN);

JC=(TP)/(TP+FN+FP)

TPR=TP/(TP+FN);% True Positive Rate (sensitivity/recall/hit rate)
FPR=FP/(FP+TN);% False Positive Rate (specificity=1-FPR)
PPV=TP/(TP+FP);%positive predictive value (precision)
r=[JC]
% r=[JC;FPR;accuracy;PPV];