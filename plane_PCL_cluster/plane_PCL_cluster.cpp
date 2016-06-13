// plane_PCL_cluster.cpp : Defines the entry point for the console application.
//
// plane_PCL_RANSAC.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <math.h>
#include <cmath>
#include <Eigen/Dense>

#include <iostream>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/mlesac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
//#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/filter.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/centroid.h>

#include <time.h>
#include <algorithm>
#include "kmeans.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#define PI 3.14159265

//const int imagewidth = 1240;
const int imagewidth = 640;
const int imageheight = 368;
const int imagesize = imagewidth * imageheight;

class kiitKFoutput
{
public:
	Eigen::Matrix3f K;
	Eigen::Matrix3f rotationMatrix_cam2world;
	Eigen::Vector3f translationVector_cam2world;
	int currentKFParentID;
	Eigen::Matrix3f rotationMatrix_cam2parent;
	Eigen::Vector3f translationVector_cam2parent;
	std::vector<float> idepth;
	std::vector<float> idepthVar;
	std::vector<float> imagefloat;//gray scale value
};
class KFoutput
{
public:
	Eigen::Matrix3f K;
	Eigen::Matrix3f rotationMatrix_cam2world;
	Eigen::Vector3f translationVector_cam2world;
	std::vector<float> idepth;
	std::vector<float> idepthVar;
	std::vector<float> imagefloat;//gray scale value
};

void KF2cloud(KFoutput KFinput, pcl::PointCloud<pcl::PointXYZI>::Ptr CloudOut, float depthVar_thres = INFINITY,
	pcl::PointCloud<pcl::PointXYZ>::Ptr CloudNoColor = nullptr) {
	for (int i = 0; i < imageheight; i++)
	{
		for (int j = 0; j < imagewidth; j++)
		{
			int idx = i*imagewidth + j;
			if (KFinput.idepthVar.at(idx) >= 0 && KFinput.imagefloat.at(idx) >= 0)
			{
				if (KFinput.idepthVar.at(idx) <= depthVar_thres)
				{
					Eigen::MatrixXf point_tmp(3, 1);
					Eigen::MatrixXf point3D_tmp(3, 1);
					point_tmp(0, 0) = j;
					point_tmp(1, 0) = i;
					point_tmp(2, 0) = 1;
					point3D_tmp = KFinput.K.inverse()*point_tmp;
					pcl::PointXYZI pointc;
					pointc.x = point3D_tmp(0, 0) / KFinput.idepth.at(idx);
					pointc.y = point3D_tmp(1, 0) / KFinput.idepth.at(idx);
					pointc.z = point3D_tmp(2, 0) / KFinput.idepth.at(idx);
					pointc.intensity = KFinput.imagefloat.at(idx);
					CloudOut->points.push_back(pointc);
					if (CloudNoColor)
					{
						pcl::PointXYZ point;
						point.x = point3D_tmp(0, 0) / KFinput.idepth.at(idx);
						point.y = point3D_tmp(1, 0) / KFinput.idepth.at(idx);
						point.z = point3D_tmp(2, 0) / KFinput.idepth.at(idx);
						CloudNoColor->points.push_back(point);
					}
				}
			}
		}
	}
	CloudOut->width = CloudOut->points.size();
	CloudOut->height = 1;
	CloudOut->is_dense = true;
	CloudOut->points.resize(CloudOut->width * CloudOut->height);
	if (CloudNoColor)
	{
		CloudNoColor->width = CloudNoColor->points.size();
		CloudNoColor->height = 1;
		CloudNoColor->is_dense = true;
		CloudNoColor->points.resize(CloudNoColor->width * CloudNoColor->height);
	}
}

KFoutput read_bin(std::string &infilename) {
	KFoutput KF;
	std::streampos cur;
	std::fstream infile(infilename, std::ios::binary | std::ios::in);
	if (!infile) {
		std::cout << "bin open error!" << std::endl;
	}
	Eigen::Matrix3f matrix_tmp;
	Eigen::Vector3f vect_tmp;
	float float_tmp;
	//while (!infile.eof()) {}
	infile.read((char*)&matrix_tmp, sizeof(Eigen::Matrix3f));
	KF.K = matrix_tmp;
	cur = infile.tellg();
	infile.read((char*)&matrix_tmp, sizeof(Eigen::Matrix3f));
	KF.rotationMatrix_cam2world = matrix_tmp;
	cur = infile.tellg();
	infile.read((char*)&vect_tmp, sizeof(Eigen::Vector3f));
	KF.translationVector_cam2world = vect_tmp;
	cur = infile.tellg();
	for (int i = 0; i < imagesize; i++)
	{
		infile.read((char*)&float_tmp, sizeof(float));
		KF.idepth.push_back(float_tmp);
	}
	for (int i = 0; i < imagesize; i++)
	{
		infile.read((char*)&float_tmp, sizeof(float));
		KF.idepthVar.push_back(float_tmp);
	}
	for (int i = 0; i < imagesize; i++)
	{
		infile.read((char*)&float_tmp, sizeof(float));
		KF.imagefloat.push_back(float_tmp);
	}
	//cur = infile.tellg();
	infile.close();
	return KF;
}

void kiitKF2cloud(kiitKFoutput KFinput, pcl::PointCloud<pcl::PointXYZI>::Ptr CloudOut, float depthVar_thres = INFINITY,
	pcl::PointCloud<pcl::PointXYZ>::Ptr CloudNoColor = nullptr) {
	for (int i = 0; i < imageheight; i++)
	{
		for (int j = 0; j < imagewidth; j++)
		{
			int idx = i*imagewidth + j;
			if (KFinput.idepthVar.at(idx) >= 0 && KFinput.imagefloat.at(idx) >= 0)
			{
				if (KFinput.idepthVar.at(idx) <= depthVar_thres)
				{
					Eigen::MatrixXf point_tmp(3, 1);
					Eigen::MatrixXf point3D_tmp(3, 1);
					point_tmp(0, 0) = j;
					point_tmp(1, 0) = i;
					point_tmp(2, 0) = 1;
					point3D_tmp = KFinput.K.inverse()*point_tmp;
					pcl::PointXYZI pointc;
					pointc.x = point3D_tmp(0, 0) / KFinput.idepth.at(idx);
					pointc.y = point3D_tmp(1, 0) / KFinput.idepth.at(idx);
					pointc.z = point3D_tmp(2, 0) / KFinput.idepth.at(idx);
					pointc.intensity = KFinput.imagefloat.at(idx);
					CloudOut->points.push_back(pointc);
					if (CloudNoColor)
					{
						pcl::PointXYZ point;
						point.x = point3D_tmp(0, 0) / KFinput.idepth.at(idx);
						point.y = point3D_tmp(1, 0) / KFinput.idepth.at(idx);
						point.z = point3D_tmp(2, 0) / KFinput.idepth.at(idx);
						CloudNoColor->points.push_back(point);
					}
				}
			}
		}
	}
	CloudOut->width = CloudOut->points.size();
	CloudOut->height = 1;
	CloudOut->is_dense = true;
	CloudOut->points.resize(CloudOut->width * CloudOut->height);
	if (CloudNoColor)
	{
		CloudNoColor->width = CloudNoColor->points.size();
		CloudNoColor->height = 1;
		CloudNoColor->is_dense = true;
		CloudNoColor->points.resize(CloudNoColor->width * CloudNoColor->height);
	}
}

kiitKFoutput read_kiitbin(std::string &infilename) {
	kiitKFoutput KF;
	std::streampos cur;
	std::fstream infile(infilename, std::ios::binary | std::ios::in);
	if (!infile) {
		std::cout << "bin open error!" << std::endl;
	}
	Eigen::Matrix3f matrix_tmp;
	Eigen::Vector3f vect_tmp;
	float float_tmp;
	infile.read((char*)&matrix_tmp, sizeof(Eigen::Matrix3f));
	KF.K = matrix_tmp;
	cur = infile.tellg();
	infile.read((char*)&matrix_tmp, sizeof(Eigen::Matrix3f));
	KF.rotationMatrix_cam2world = matrix_tmp;
	cur = infile.tellg();
	infile.read((char*)&vect_tmp, sizeof(Eigen::Vector3f));
	KF.translationVector_cam2world = vect_tmp;
	cur = infile.tellg();

	int int_tmp;
	infile.read((char*)&int_tmp, sizeof(int));
	KF.currentKFParentID = int_tmp;
	cur = infile.tellg();
	infile.read((char*)&matrix_tmp, sizeof(Eigen::Matrix3f));
	KF.rotationMatrix_cam2parent = matrix_tmp;
	cur = infile.tellg();
	infile.read((char*)&vect_tmp, sizeof(Eigen::Vector3f));
	KF.translationVector_cam2parent = vect_tmp;
	cur = infile.tellg();

	for (int i = 0; i < imagesize; i++)
	{
		infile.read((char*)&float_tmp, sizeof(float));
		KF.idepth.push_back(float_tmp);
	}
	for (int i = 0; i < imagesize; i++)
	{
		infile.read((char*)&float_tmp, sizeof(float));
		KF.idepthVar.push_back(float_tmp);
	}
	for (int i = 0; i < imagesize; i++)
	{
		infile.read((char*)&float_tmp, sizeof(float));
		KF.imagefloat.push_back(float_tmp);
	}
	//cur = infile.tellg();
	infile.close();
	return KF;
}

boost::shared_ptr<pcl::visualization::PCLVisualizer>
simpleVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
	// --------------------------------------------
	// ----- 3D viewer and add point cloud-----
	// --------------------------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
	//viewer->addCoordinateSystem (1.0, "global");
	viewer->initCameraParameters();
	return (viewer);
}

boost::shared_ptr<pcl::visualization::CloudViewer>
simpleVis(pcl::PointCloud<pcl::PointXYZI>::ConstPtr cloud)
{
	// --------------------------------------------
	// ----- 3D viewer and add point cloud-----
	// --------------------------------------------
	boost::shared_ptr<pcl::visualization::CloudViewer> viewer(new pcl::visualization::CloudViewer("3D Viewer"));
	viewer->showCloud(cloud, "sample cloud");
	//viewer->addCoordinateSystem (1.0, "global");
	return (viewer);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr downsample(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float leafsize[3]) {
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudout(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::VoxelGrid<pcl::PointXYZ> sor;
	sor.setInputCloud(cloud);
	sor.setLeafSize(leafsize[0], leafsize[1], leafsize[2]);
	sor.filter(*cloudout);
	return cloudout;
}
pcl::PointCloud<pcl::PointXYZI>::Ptr downsamplecolor(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, float leafsize[3]) {
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloudout(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::VoxelGrid<pcl::PointXYZI> sor;
	sor.setInputCloud(cloud);
	sor.setLeafSize(leafsize[0], leafsize[1], leafsize[2]);
	sor.filter(*cloudout);
	return cloudout;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr RadiusFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr inputcloud, double radius, int num) {
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_radius(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
	// build the filter
	outrem.setInputCloud(inputcloud);
	outrem.setRadiusSearch(radius);
	outrem.setMinNeighborsInRadius(num);
	// apply filter
	outrem.filter(*cloud_filtered_radius);
	return cloud_filtered_radius;
}

int
main()
{
	clock_t start_t, finish_t; //CLOCKS_PER_SEC

	pcl::PointCloud<pcl::PointXYZI>::Ptr cloudcolor(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr final(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXY>::Ptr plane_point(new pcl::PointCloud<pcl::PointXY>);
	std::vector<int> inliers1;

	float inlier_threshold = 0.02;
	int iter_max = 100;
	float depthVar_thres = 0.05;

	//down sample
	bool dodownsample = false;
	float leafsize[3] = { 0.1f, 0.1f, 0.1f };
	//radius filter
	bool doradiusfilter = false;
	float radius_plane = 0.03;
	int radius_plane_num = 10;
	int num_clusters = 3; //num_clusters <= 5

	std::string filename1 = "E:\\kiitbindata\\51.bin";

	//kiitKFoutput KF1;
	//KF1 = read_kiitbin(filename1);
	//kiitKF2cloud(KF1, cloudcolor, depthVar_thres, cloud);

	filename1 = "E:\\stereoslamtestdata\\output\\goproslambindata\\1800.bin";
	KFoutput KF1;
	KF1 = read_bin(filename1);
	KF2cloud(KF1, cloudcolor, depthVar_thres, cloud);

	Eigen::Matrix3f R1_cam2world = KF1.rotationMatrix_cam2world;
	Eigen::Matrix3Xf T1_cam2world(3, 1);
	T1_cam2world = KF1.translationVector_cam2world;
	std::vector<float> depth1;
	std::vector<float> depthVar1;
	std::vector<float> color1;
	Eigen::Matrix3f K1;
	Eigen::Matrix3f K1_inverse;
	std::cout << "total points: " << KF1.imagefloat.size() << std::endl;
	int num_valid_points1 = 0;
	for (int i = 0; i < imagesize; i++)
	{
		depth1.push_back(KF1.idepth.at(i));
		depthVar1.push_back(KF1.idepthVar.at(i));
		color1.push_back(KF1.imagefloat.at(i));
	}
	K1 = KF1.K;
	K1_inverse = K1.inverse();

	std::cout << "valid points: " << cloudcolor->points.size() << std::endl;

	if (dodownsample)
	{
		start_t = clock();
		cloud = downsample(cloud, leafsize);
		finish_t = clock();
		std::cout << "down sample time: " << double(finish_t - start_t) / CLOCKS_PER_SEC << std::endl;
		cloudcolor = downsamplecolor(cloudcolor, leafsize);
	}
	// created RandomSampleConsensus object and compute the appropriated model Sphere and Plane
	//pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr
	//	model_s(new pcl::SampleConsensusModelSphere<pcl::PointXYZ>(cloud));

	start_t = clock();
	Eigen::VectorXf ransac_plane_coef;
	std::vector<int> ransac_plane_model;
	pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr
		model_p(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud));
	pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_p);
	ransac.setDistanceThreshold(inlier_threshold);
	//ransac.setSampleConsensusModel();
	ransac.setMaxIterations(iter_max);
	ransac.computeModel();
	ransac.getInliers(inliers1);
	finish_t = clock();
	ransac.getModelCoefficients(ransac_plane_coef); 
	ransac.getModel(ransac_plane_model);

	std::cout <<"ransac plane coefficient: " << std::endl << ransac_plane_coef(0) << std::endl << ransac_plane_coef(1) << std::endl << ransac_plane_coef(2) << std::endl << ransac_plane_coef(3) << std::endl << std::endl;
	std::cout << "ransac inlier_threshold: " << inlier_threshold << " #points input :" << cloud->points.size() <<
		" #inliers: " << inliers1.size() << " time: " << double(finish_t - start_t) / CLOCKS_PER_SEC << std::endl;

	float plane_a = ransac_plane_coef(0);
	float plane_b = ransac_plane_coef(1);
	float plane_c = ransac_plane_coef(2);
	float plane_d = ransac_plane_coef(3);
	float normalized = sqrt(plane_a * plane_a + plane_b * plane_b + plane_c * plane_c);
	std::vector<float> normal_vect = { plane_a / normalized, plane_b / normalized, plane_c / normalized };
	normalized = sqrt(plane_a * plane_a + plane_c * plane_c);
	std::vector<float> rot_vect = { -plane_c / normalized, 0, plane_a / normalized };
	float theta = acos(normal_vect[1]) * 180.0 / PI;
	//rotate plane to horizontal(y=Const.)
	Eigen::Matrix3f plane_rot;
	plane_rot(0, 0) = rot_vect[0] * rot_vect[0] + (1 - rot_vect[0] * rot_vect[0])*cos(theta);
	plane_rot(0, 1) = -rot_vect[2] * sin(theta) + (1 - cos(theta))*rot_vect[0] * rot_vect[1];
	plane_rot(0, 2) = rot_vect[1] * sin(theta) + (1 - cos(theta))*rot_vect[0] * rot_vect[2];
	plane_rot(1, 0) = rot_vect[2] * sin(theta) + (1 - cos(theta))*rot_vect[0] * rot_vect[1];
	plane_rot(1, 1) = rot_vect[1] * rot_vect[1] + (1 - rot_vect[1] * rot_vect[1])*cos(theta);
	plane_rot(1, 2) = -rot_vect[0] * sin(theta) + (1 - cos(theta))*rot_vect[1] * rot_vect[2];
	plane_rot(2, 0) = -rot_vect[1] * sin(theta) + (1 - cos(theta))*rot_vect[0] * rot_vect[2];
	plane_rot(2, 1) = rot_vect[0] * sin(theta) + (1 - cos(theta))*rot_vect[2] * rot_vect[1];
	plane_rot(2, 2) = rot_vect[2] * rot_vect[2] + (1 - rot_vect[2] * rot_vect[2])*cos(theta);

	//get outlier
	std::vector<int> outlier;
	pcl::PointCloud<pcl::PointXYZI>::Ptr outlier1color(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr outlier1(new pcl::PointCloud<pcl::PointXYZ>);
	std::sort(inliers1.begin(), inliers1.end());
	auto inlier_iter = inliers1.begin();
	for (int i = 0; i < cloud->points.size(); i++)
	{
		if (inlier_iter != inliers1.end() && i == *inlier_iter)
		{
			inlier_iter = inlier_iter + 1;
		}
		else
		{
			outlier1->points.push_back(cloud->points.at(i));
			outlier1color->points.push_back(cloudcolor->points.at(i));
		}
	}

	// copies all inliers of the model computed to another PointCloud
	pcl::copyPointCloud<pcl::PointXYZ>(*cloud, inliers1, *final);
	for (int i = 0; i < cloud->points.size(); i++)
	{
		cloudcolor->points.at(i).intensity = 200;
	}
	for (int i = 0; i < inliers1.size(); i++)
	{
		cloudcolor->points.at(inliers1.at(i)).intensity = 10;
	}
	if (doradiusfilter)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
		tmp = RadiusFilter(final, radius_plane, radius_plane_num);
		final->clear();
		final = tmp;
	}

	for (int i = 0; i < final->points.size(); i++)
	{
		pcl::PointXY point_tmp;
		point_tmp.x = final->points.at(i).x;
		point_tmp.y = final->points.at(i).z;
		plane_point->push_back(point_tmp);
	}

	/*K-means BEGIN*/
	int pointsize = plane_point->points.size(); //Number of samples
	const int dim = 2;   //Dimension of feature
	const int KK = num_clusters; //Cluster number
	std::vector<double> plane_data;
	for (int i = 0; i<pointsize; i++)
	{
		plane_data.push_back(plane_point->points.at(i).x);
		plane_data.push_back(plane_point->points.at(i).y);
	}
	KMeans* kmeans = new KMeans(dim, KK);
	std::vector<int> labels;
	kmeans->SetInitMode(KMeans::InitUniform);
	labels = kmeans->Cluster(plane_data);
	std::vector<int> cluster0;
	std::vector<int> cluster1;
	std::vector<int> cluster2;
	std::vector<int> cluster3;
	std::vector<int> cluster4;
	for (int i = 0; i<pointsize; i++)
	{
		if (labels[i] == 0)
		{
			cluster0.push_back(i);
		}
		if (labels[i] == 1)
		{
			cluster1.push_back(i);
		}
		if (labels[i] == 2)
		{
			cluster2.push_back(i);
		}
		if (labels[i] == 3)
		{
			cluster3.push_back(i);
		}
		if (labels[i] == 4)
		{
			cluster4.push_back(i);
		}
	}

	//find largest cluster
	pcl::PointCloud<pcl::PointXYZ>::Ptr largestcluster(new pcl::PointCloud<pcl::PointXYZ>);
	std::vector<cv::Point2f> cv_cluster_points2D;
	// Create and accumulate points
	pcl::CentroidPoint<pcl::PointXYZ> centroid;
	if (cluster0.size() >= cluster1.size() && cluster0.size() >= cluster2.size() && cluster0.size() >= cluster3.size() && cluster0.size() >= cluster4.size())
	{
		largestcluster->width = cluster0.size();
		largestcluster->height = 1;
		for (int i = 0; i < cluster0.size(); i++)
		{
			pcl::PointXYZ tmp;
			tmp.x = plane_point->points.at(cluster0.at(i)).x;
			tmp.z = plane_point->points.at(cluster0.at(i)).y;
			tmp.y = - plane_d / plane_b;
			largestcluster->push_back(tmp);
			centroid.add(tmp);
			cv_cluster_points2D.push_back(cv::Point2f(tmp.x, tmp.z));
		}
		
	} 
	else if (cluster1.size() >= cluster0.size() && cluster1.size() >= cluster2.size() && cluster1.size() >= cluster3.size() && cluster1.size() >= cluster4.size())
	{
		largestcluster->width = cluster1.size();
		largestcluster->height = 1;
		for (int i = 0; i < cluster1.size(); i++)
		{
			pcl::PointXYZ tmp;
			tmp.x = plane_point->points.at(cluster1.at(i)).x;
			tmp.z = plane_point->points.at(cluster1.at(i)).y;
			tmp.y = -plane_d / plane_b;
			largestcluster->push_back(tmp);
			centroid.add(tmp);
			cv_cluster_points2D.push_back(cv::Point2f(tmp.x, tmp.z));
		}
	} 
	else if (cluster2.size() >= cluster0.size() && cluster2.size() >= cluster1.size() && cluster2.size() >= cluster3.size() && cluster2.size() >= cluster4.size())
	{
		largestcluster->width = cluster2.size();
		largestcluster->height = 1;
		for (int i = 0; i < cluster2.size(); i++)
		{
			pcl::PointXYZ tmp; 
			tmp.x = plane_point->points.at(cluster2.at(i)).x;
			tmp.z = plane_point->points.at(cluster2.at(i)).y;
			tmp.y = -plane_d / plane_b;
			largestcluster->push_back(tmp);
			centroid.add(tmp);
			cv_cluster_points2D.push_back(cv::Point2f(tmp.x, tmp.z));
		}
	}
	else if (cluster3.size() >= cluster0.size() && cluster3.size() >= cluster1.size() && cluster3.size() >= cluster2.size() && cluster3.size() >= cluster4.size())
	{
		largestcluster->width = cluster3.size();
		largestcluster->height = 1;
		for (int i = 0; i < cluster3.size(); i++)
		{
			pcl::PointXYZ tmp;
			tmp.x = plane_point->points.at(cluster3.at(i)).x;
			tmp.z = plane_point->points.at(cluster3.at(i)).y;
			tmp.y = -plane_d / plane_b;
			largestcluster->push_back(tmp);
			centroid.add(tmp);
			cv_cluster_points2D.push_back(cv::Point2f(tmp.x, tmp.z));
		}
	}
	else
	{
		largestcluster->width = cluster4.size();
		largestcluster->height = 1;
		for (int i = 0; i < cluster4.size(); i++)
		{
			pcl::PointXYZ tmp;
			tmp.x = plane_point->points.at(cluster4.at(i)).x;
			tmp.z = plane_point->points.at(cluster4.at(i)).y;
			tmp.y = -plane_d / plane_b;
			largestcluster->push_back(tmp);
			centroid.add(tmp);
			cv_cluster_points2D.push_back(cv::Point2f(tmp.x, tmp.z));
		}
	}
	pcl::PointXYZ cluster_center;
	centroid.get(cluster_center);
	//std::map<int, pcl::PointXY> center;
	//for (int i = 0; i<KK; i++)
	//{
	//	int idx = rand() % plane_point->points.size();
	//	pcl::PointXY point_tmp = plane_point->points.at(idx);
	//	center.insert_or_assign(i, point_tmp);
	//}
	//for (int i = 0; i < plane_point->points.size(); i++)
	//{
	//	float dist0 = std::abs(center.at(0).x - plane_point->points.at(i).x) + std::abs(center.at(0).y - plane_point->points.at(i).y);
	//	float dist1 = std::abs(center.at(1).x - plane_point->points.at(i).x) + std::abs(center.at(1).y - plane_point->points.at(i).y);
	//	float dist2 = std::abs(center.at(2).x - plane_point->points.at(i).x) + std::abs(center.at(2).y - plane_point->points.at(i).y);
	//}
	//std::vector<cv::RotatedRect> minRect(contours.size());
	/*K-means END*/

	/*Get rectangular points BEGIN*/
	cv::RotatedRect rRect = cv::minAreaRect(cv_cluster_points2D);
	cv::Point2f cv_cluster_rect[4];//x&z coordinates, y = - plane_d / plane_b
	rRect.points(cv_cluster_rect);
	//cv::Rect brect = rRect.boundingRect();
	/*Get rectangular points END*/

	/*SEGMENT method*/
	//pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	//pcl::PointIndices::Ptr inliers_seg(new pcl::PointIndices);
	//start_t = clock();
	//// Create the segmentation object
	//pcl::SACSegmentation<pcl::PointXYZ> seg;
	//// Optional
	//seg.setOptimizeCoefficients(true);
	//// Mandatory
	//seg.setModelType(pcl::SACMODEL_PLANE);
	//seg.setMethodType(pcl::SAC_MLESAC);
	//*  const static int SAC_RANSAC  = 0;
	//const static int SAC_LMEDS   = 1;
	//const static int SAC_MSAC    = 2;
	//const static int SAC_RRANSAC = 3;
	//const static int SAC_RMSAC   = 4;
	//const static int SAC_MLESAC  = 5;
	//const static int SAC_PROSAC  = 6;*/
	//seg.setDistanceThreshold(inlier_threshold);
	//seg.setInputCloud(cloud);
	//seg.segment(*inliers_seg, *coefficients);
	//finish_t = clock();
	//std::cout << "seg inlier_threshold: " << inlier_threshold << " #points input :" << cloud->points.size() <<
	//	" #inliers: " << inliers_seg->indices.size() << " seg time: " << double(finish_t - start_t) / CLOCKS_PER_SEC << std::endl;

	boost::shared_ptr<pcl::visualization::CloudViewer> viewer;
	viewer = simpleVis(cloudcolor);
	while (!viewer->wasStopped())
	{

	}

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewerp;
	viewerp = simpleVis(final);
	while (!viewerp->wasStopped())
	{
		viewerp->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewerl;
	viewerl = simpleVis(largestcluster);
	while (!viewerl->wasStopped())
	{
		viewerl->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewerf;
	viewerf = simpleVis(cloud);
	pcl::PointXYZ pointa;
	pcl::PointXYZ pointb;
	pcl::PointXYZ pointc;
	pcl::PointXYZ pointd;
	pointa.x = cv_cluster_rect[0].x; 
	pointa.y = -plane_d / plane_b;
	pointa.z = cv_cluster_rect[0].y;
	pointb.x = cv_cluster_rect[1].x;
	pointb.y = -plane_d / plane_b;
	pointb.z = cv_cluster_rect[1].y;
	pointc.x = cv_cluster_rect[2].x;
	pointc.y = -plane_d / plane_b;
	pointc.z = cv_cluster_rect[2].y;
	pointd.x = cv_cluster_rect[3].x;
	pointd.y = -plane_d / plane_b;
	pointd.z = cv_cluster_rect[3].y;
	viewerf->addLine(pointa, pointb, "1");
	viewerf->addLine(pointb, pointc, "2");
	viewerf->addLine(pointc, pointd, "3");
	viewerf->addLine(pointd, pointa, "4");
	while (!viewerf->wasStopped())
	{
		viewerf->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

	return 0;
}