clc
clear
close all
addpath('D:\qiukev\MATLAB_Dateien','D:\qiukev\RandLA-Net-Hessigheim')

load('D:\waglar\pytorch_datasets\potsdamraw\processed\imgs\0_0_0.mat')
im1 = load('D:\waglar\pytorch_datasets\potsdamraw\processed\imgs\0_0_0.mat')
im2 = load('D:\waglar\pytorch_datasets\potsdamraw\processed\imgs\0_0_1.mat')
im3 = load('D:\waglar\pytorch_datasets\potsdamraw\processed\imgs\0_0_2.mat')
im4 = load('D:\waglar\pytorch_datasets\potsdamraw\processed\imgs\0_0_3.mat')
display_input_gr_axkt_18([],[],1,4,jet(128),im1.img(:,:,1:3),im2.img(:,:,1:3),im3.img(:,:,1:3),im4.img(:,:,1:3));

