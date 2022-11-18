 /********************************************************
  * NetPred.hpp
  * 使用网络预测地形
  * 
  * Author: Born Chow
  * Date: 2021.09.29 
  * modify: 2021.12.06
  * 
  * How to use:
  * 申明一个类用于点云模型预测
  * 1. 申明
  * shared_ptr<NetPred> net_pred_ = make_shared<NetPred>();
  * 2. 初始化
  * net_pred_->InitParam(sample_point_num_, box_stride_, box_kernel_, net_model_dir_);  //适用于point model的参数初始化
  * 3. 调用推理
  * net_pred_->PointPred(pc_in_body, process_pc_in_body, process_pc_pass); //调用point model
  * 
  * ///////
  * 
  * 申明一个类用于CNN模型预测 20211206
  * 1. 申明
  * shared_ptr<NetPred> net_pred_ = make_shared<NetPred>();
  * 2. 初始化
  * net_pred_->InitParam(int img_size, float map_res, float box_stride, float box_kernel, string model_dir) //适用于CNN模型
  * 3. 调用推理
  * net_pred_->CNNPred(pc_in_body, process_pc_in_body, process_pc_pass); //调用CNN model
  * 
 ******************************************************/


#include "torch/script.h"
#include "torch/torch.h"
#include <navi_pred/common.h> // struct Bbox_AABB
#include <navi_pred/tictoc.h>


// #define SHOW_POINT
// #define SHOW_POINT_BOX


class NetPred
{
private:

    // PcsToTensor 中的参数
    int sample_num_;

    // CreatBboxs 函数中的参数 
    float stride_;
    float kernel_;

    // model name torch的模型位置
    string model_dir_;

    //PcsToElevaToTensor 中的参数
    int img_size_;
    float map_res_; 


    torch::jit::script::Module model_;



private:
    std::vector<Bbox_AABB> CreatBboxs(Bbox_AABB aabb, float stride, float kernel);
    torch::Tensor PcsToTensor(pcl::PointCloud<PointType>::Ptr cloud, int sample_points);

    torch::Tensor PcsToElevaToTensor(pcl::PointCloud<PointType>::Ptr pc, int img_size, float res);
    
    
    
public:
    NetPred(int sample_num, float box_stride, float box_kernel, string model_dir);
    
    void PointPred(pcl::PointCloud<PointType>::Ptr cloud, 
                    pcl::PointCloud<PointType>::Ptr &cloud_pred, 
                    pcl::PointCloud<PointType>::Ptr &cloud_pred_pass
                    );

    void PointPred(pcl::PointCloud<PointType>::Ptr cloud, 
                    std::vector<pcl::PointCloud<PointType>::Ptr > sample_point_pts,
                    pcl::PointCloud<PointType>::Ptr &cloud_pred, 
                    pcl::PointCloud<PointType>::Ptr &cloud_pred_pass,
                    std::vector<int> &sample_point_pred
                    );
    
    void PointPred(std::vector<pcl::PointCloud<PointType>::Ptr > sample_point_pts,
                    std::vector<int> &sample_point_pred
                    );

    void CNNPred(pcl::PointCloud<PointType>::Ptr cloud, 
                pcl::PointCloud<PointType>::Ptr &cloud_pred, 
                pcl::PointCloud<PointType>::Ptr &cloud_pred_pass);

    void InitParam(int sample_num, float box_stride, float box_kernel, string model_dir);

    void InitParam(int img_size, float map_res, float box_stride, float box_kernel, string model_dir);
    NetPred();
    ~NetPred();
};

NetPred::NetPred(){
 
    std::cout << "=== [NetPred] ==> net pred start" << std::endl;
}

NetPred::NetPred(int sample_num, float box_stride, float box_kernel, string model_dir) :
sample_num_(sample_num),
stride_(box_kernel),
kernel_(box_kernel),
model_dir_(model_dir)
{
    model_ = torch::jit::load(model_dir_);
    model_.to(torch::kCUDA);
    model_.eval();
    std::cout << "=== [NetPred] ==> net pred start" << std::endl;
    std::cout << "=== [NetPred] ==> load net model" << model_dir_ << std::endl;
}

NetPred::~NetPred()
{

}


/**
 * @brief InitParam 初始化参数  --> 这里是用于点云模型预测的参数
 * @param sample_num   【输入】 对每一个需要输入网络的点云进行重采样后的点数
 * @param box_stride   【输入】 滑窗预测时的步长
 * @param box_kernel   【输入】 滑窗预测时滑窗的大小
 * @param model_dir    【输入】 点云网络模型的路径 xxx.pt文件
 */
void NetPred::InitParam(int sample_num, float box_stride, float box_kernel, string model_dir){

    sample_num_ = sample_num;
    stride_ = box_stride;
    kernel_ = box_kernel;
    model_dir_ = model_dir;

    // load model
    torch::NoGradGuard no_grad;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        model_ = torch::jit::load(model_dir_);
        //gpu optimize
        model_.to(torch::kCUDA);
        model_.eval();
        std::cout << "=== [NetPred] ==> load net model" << model_dir_ << std::endl;
        std::cout << "=== [NetPred] ==> init finished " << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "=== [NetPred] ==> error loading the model " << model_dir_ << std::endl;
        return;
    }
}


/**
 * @brief InitParam 初始化参数  --> 这里是用于CNN模型预测的参数
 * @param img_size     【输入】 输入网络的高程图大小 论文中为20*20
 * @param map_res      【输入】 从点云制作高程图的过程中，采用的octomap的分辨率,这直接影响高程图的大小
 * @param box_stride   【输入】 滑窗预测时的步长
 * @param box_kernel   【输入】 滑窗预测时滑窗的大小
 * @param model_dir    【输入】 点云网络模型的路径 xxx.pt文件
 */
void NetPred::InitParam(int img_size, float map_res, float box_stride, float box_kernel, string model_dir){

    img_size_ = img_size;
    map_res_ = map_res;
    stride_ = box_stride;
    kernel_ = box_kernel;
    model_dir_ = model_dir;

    // load model
    torch::NoGradGuard no_grad;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        model_ = torch::jit::load(model_dir_);
        //gpu optimize
        model_.to(torch::kCUDA);
        model_.eval();
        std::cout << "=== [NetPred] ==> load net model" << model_dir_ << std::endl;
        std::cout << "=== [NetPred] ==> init finished " << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "=== [NetPred] ==> error loading the model " << model_dir_ << std::endl;
        return;
    }

}



/**
 * @brief PcsToElevaToTensor 将PCL中的pointcloud数据格式点云转换为octomap，然后再转换为torch中的tensor格式
 * @param pc          输入点云指针
 * @param img_size    生产高程图的尺寸 img_size*img_size
 * @param res         octomap中采用的分辨率
 * @return torch::tensor
 */
torch::Tensor NetPred::PcsToElevaToTensor(pcl::PointCloud<PointType>::Ptr pc, int img_size, float res){

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*pc, *pc, indices);

    shared_ptr<OcTreeT> map;
    map.reset(new OcTreeT(res));

    //加载pc
    for(auto p:pc->points){
        map->updateNode(octomap::point3d(p.x, p.y, p.z), true);
    }

    unsigned max_TreeDepth = map->getTreeDepth();

    double size_x, size_y, size_z;
    map -> getMetricSize(size_x, size_y, size_z); //获得 octomap 的 size

    double minX, minY, minZ;
    double maxX, maxY, maxZ;
    map->getMetricMin(minX, minY, minZ);
    map->getMetricMax(maxX, maxY, maxZ);
    //cout << "Map min: " << minX << " " << minY << "  " << minZ << " Max : " << maxX << " " << maxY << " " << maxZ << std::endl;
    cv::Mat img(img_size, img_size, CV_16SC1, cv::Scalar(0));  //根据小车的roi区域每次创建的高程图大小都是一致的

    if (ceil(size_x/res) < img.rows || ceil(size_y/res) < img.cols){
        minY = minY < -size_y /2 ? minY : -size_y /2;
        minX = minX < -size_x /2 ? minX : -size_x /2;
        //cout << "min adjust: " << minX << " " << minY << "  " << minZ << " Max : " << maxX << " " << maxY << " " << maxZ << std::endl;
    }

    //cout << "img size: " << img.rows << " " << img.cols << endl;
    //cout << "map size: " << size_x << " " << size_y << " " << size_z  << " "<<  ceil(size_x/res) <<" " << ceil(size_y/res) << " " << size_z/res << " " << endl;

    int i = 0;
    double img_x = 0, img_y = 0;  //图像坐标系
    int pixel_x = 0; int pixel_y = 0; //像素坐标系
    //   图像坐标系定义
    //      -------->  y
    //      |
    //      |
    //      |
    //   x  V
    //遍历octomap
    for (OcTreeT::iterator it = map->begin(max_TreeDepth), end = map->end(); it != end; ++it)
    {
        if (it.getDepth() == max_TreeDepth){ //说明是最小的节点
            i++;
            img_x = it.getX();
            img_y = it.getY();
            pixel_x = floor((img_x - minX)/res);
            pixel_y = floor((img_y - minY)/res);

            if(pixel_x < img.rows && pixel_y < img.cols && pixel_y >= 0 && pixel_x >= 0) {
               // std::cout << i << " img_P " << it.getX() << "," << it.getY() << " : " << it.getZ() * 100 <<
               //         " min:  " << minX << " , " << minY << " , " <<
               //          " --> " << pixel_x << " " << pixel_y << std::endl;
                img.at<short>(pixel_x, pixel_y) = it.getZ() * 100; //高程的单位换为cm
            }
        }
    }


    // std::cout << "img vector " << std::endl;
    std::vector<float> img_1d;
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            float v = img.at<short>(i, j);
            //std::cout << v << " ";
            img_1d.push_back(v);
        }
    }
    //std::cout<<std::endl;

    torch::Tensor tensor_image = torch::from_blob(img_1d.data(), {img.rows, img.cols, 1}, torch::kFloat);

    tensor_image = tensor_image.permute({2, 0, 1});

    //std::cout << tensor_image << std::endl;
    //std::cout << tensor_image.sizes() << std::endl;

    return tensor_image;


}


/**
 * @brief CNNPred 预测点云
 *              这里将局部点云(10M*10M)采用滑动窗口的方式分割为网络需要的点云大小(2M*2M),
 *              然后采用ocotmap生成高程图进行推理预测
 * 　　　　　　　　最后将预测的结果写入新点云的点云强度中, 所以强度越大, 危险性越高
 * @param cloud        【输入】    输入点云指针, 这里的点云需要是以小车为中心的(小车前进路线上的)某块局部点云(10M*10M或其他大小)
 * @param cloud_pred　 【返回值】   带有预测结果的点云指针 
 * @param cloud_pred_pass  【返回值】 预测结果为通过的点云指针 
 */
void NetPred::CNNPred(pcl::PointCloud<PointType>::Ptr cloud, pcl::PointCloud<PointType>::Ptr &cloud_pred, pcl::PointCloud<PointType>::Ptr &cloud_pred_pass){

    #ifdef SHOW_POINT
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    #endif

    //获取大场景点云的小包围盒集合
    pcl::MomentOfInertiaEstimation<PointType> feature_extractor;
    feature_extractor.setInputCloud(cloud);
    feature_extractor.compute();
    Bbox_AABB aabb;
    feature_extractor.getAABB(aabb.min_point, aabb.max_point);
    std::cout << "min: " << aabb.min_point << std::endl;
    std::cout << "max: " << aabb.max_point << std::endl;
    std::vector<Bbox_AABB> bboxs;
    bboxs = CreatBboxs(aabb, stride_, kernel_);
    int batch_size = bboxs.size();
    std::cout << "size: " << bboxs.size() << std::endl;


    // 拼接成tensor
    torch::Tensor com_pts_tensor = torch::ones({batch_size, 1, img_size_, img_size_});  //BNWH
    for(int i = 0; i < bboxs.size(); i++){

        #ifdef SHOW_POINT_BOX
        // 显示Bbox
        std::cout << "---- " << i << "------" <<std::endl;
        std::cout << "min : " << bboxs[i].min_point << std::endl;
        std::cout << "max : " << bboxs[i].max_point << std::endl;
        viewer->addCube(bboxs[i].min_point.x,
                        bboxs[i].max_point.x,
                        bboxs[i].min_point.y,
                        bboxs[i].max_point.y,
                        bboxs[i].min_point.z,
                        bboxs[i].max_point.z,
                        0.1*i, 0.05*i, 0.0, to_string(i));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                            0.1, to_string(i));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
                                            4, to_string(i));
        #endif

        std::vector<int> indices;
        Eigen::Vector4f min_point(bboxs[i].min_point.x, bboxs[i].min_point.y, bboxs[i].min_point.z, 1);
        Eigen::Vector4f max_point(bboxs[i].max_point.x, bboxs[i].max_point.y, bboxs[i].max_point.z, 1);
        pcl::getPointsInBox(*cloud, min_point, max_point, indices);
        if(indices.size() == 0){
            continue;
        }

        pcl::PointCloud<PointType>::Ptr cloud_batch (new pcl::PointCloud<PointType>);
        pcl::copyPointCloud(*cloud, indices, *cloud_batch);

        torch::Tensor pts_batch_tensor = PcsToElevaToTensor(cloud_batch, img_size_, map_res_).to(torch::kFloat16); // 1 * img_size_ * img_size_
        //std::cout << pts_batch_tensor << std::endl;

        // 需要将 pts_batch_tensor 中的inf值设为0
        // 20211206 如果输入tensor中有inf则会出现nan的预测值
        // https://discuss.pytorch.org/t/how-to-set-inf-in-tensor-variable-to-0/10235
        // https://blog.csdn.net/qq_39463175/article/details/108433103
        pts_batch_tensor = torch::where(torch::isinf(pts_batch_tensor), torch::full_like(pts_batch_tensor, 0), pts_batch_tensor);
        //std::cout << "-----------" <<std::endl;
        //std::cout << pts_batch_tensor << std::endl;
        com_pts_tensor.index({i, "..."}) = pts_batch_tensor;

    }

    //std::cout << "---com--" <<std::endl;
    com_pts_tensor = com_pts_tensor.to(torch::kFloat);

    //std::cout << com_pts_tensor.index({0, "..."}) << std::endl;
    //std::cout << com_pts_tensor.index({0, "..."}).sizes() <<std::endl;

    std::cout << "com size: " << com_pts_tensor.sizes() << std::endl;
    // 输入的格式是 batch_size * channel * H * W
    torch::Tensor inputs = com_pts_tensor.to(torch::kCUDA);
    std::cout << "input size: " << com_pts_tensor.sizes() << std::endl;

    // 前向推理
    torch::NoGradGuard no_grad;
    model_ = torch::jit::load(model_dir_);
    model_.to(torch::kCUDA);
    model_.eval();

    // 20211206 如果输入tensor中有inf则会出现nan的预测值
    at::Tensor output = model_.forward({inputs}).toTensor();
    //std::cout << output << std::endl;

    auto pred = output.argmax(1).to(torch::kCPU);
    //std::cout <<"pred: " << pred << std::endl;

    // 生成返回点云
    pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_pred_temp (new pcl::PointCloud<pcl::PointXYZHSV> ());
    pcl::copyPointCloud(*cloud, *cloud_pred_temp);
    pcl::copyPointCloud(*cloud, *cloud_pred);

    for(int i=0; i<cloud_pred_temp->points.size(); i++){
        cloud_pred_temp->points[i].h = 0; //用来表征总共的预测次数
        cloud_pred_temp->points[i].s = 0; //用来表征被预测为不可行的次数
        cloud_pred->points[i].intensity = 0;
    }

    // 将预测结果赋值到点云强度上
    for(int i = 0; i < bboxs.size(); i++){
        std::vector<int> indices;
        Eigen::Vector4f min_point(bboxs[i].min_point.x, bboxs[i].min_point.y, bboxs[i].min_point.z, 1);
        Eigen::Vector4f max_point(bboxs[i].max_point.x, bboxs[i].max_point.y, bboxs[i].max_point.z, 1);
        pcl::getPointsInBox(*cloud_pred, min_point, max_point, indices);
        if(pred[i].item().toInt() == 1){
            for (int j = 0; j < indices.size(); j++) {
                cloud_pred_temp->points[indices[j]].h += 1;
                cloud_pred_temp->points[indices[j]].s += 1;
            }
        }

        if(pred[i].item().toInt() == 0){
            for (int j = 0; j < indices.size(); j++) {
                cloud_pred_temp->points[indices[j]].h += 1;
            }
        }
    }

    for (int i = 0; i < cloud_pred->points.size(); i++)
    {
        // std::cout << "s: " << cloud_pred_temp->points[i].s << " h " << cloud_pred_temp->points[i].h << std::endl;
        float pro = 10.0 * cloud_pred_temp->points[i].s / cloud_pred_temp->points[i].h; // 放缩到0-10 之间
        cloud_pred->points[i].intensity = pro;
        if (pro < 5)
        {
            cloud_pred_pass->points.push_back(cloud_pred->points[i]);
        }
        
    } 
    

    #ifdef SHOW_POINT
    pcl::visualization::PointCloudColorHandlerGenericField<PointType> fildColor(cloud_pred, "intensity"); // 按照 intensity 强度字段进行渲染
    viewer->setBackgroundColor(1, 1, 1);
    viewer->addPointCloud<PointType>(cloud_pred, fildColor, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3.0, "sample cloud"); // 设置点云大小
    viewer->addCoordinateSystem(1.0);

    while (!viewer->wasStopped()){
        viewer->spinOnce();
    }
    #endif

}


/**
 * @brief PointPred 预测点云  --> 这里的输入是10*10的local map和sample_trj采样到的点云
 *              这里将局部点云采用滑动窗口的方式分割为网络需要的点云大小(2M*2M)
 * 　　　　　　　　然后将预测的结果写入新点云的点云强度中, 所以强度越大, 危险性越高
 * @param cloud            【输入】 输入点云指针, 这里的点云需要是以小车为中心的(小车前进路线上的)某块局部点云(10M*10M或其他大小)
 * @param sample_point_pts 【输入】 采样轨迹点所采样到的点云
 * @param cloud_pred　    带有预测结果的点云指针 [返回值]
 * @param cloud_pred_pass 预测结果为通过的点云指针 [返回值]
 * @param sample_point_pred 【返回值】 采样轨迹点的预测结果
 */
void NetPred::PointPred(pcl::PointCloud<PointType>::Ptr cloud, 
                        std::vector<pcl::PointCloud<PointType>::Ptr > sample_point_pts,
                        pcl::PointCloud<PointType>::Ptr &cloud_pred, 
                        pcl::PointCloud<PointType>::Ptr &cloud_pred_pass,
                        std::vector<int> &sample_point_pred){
    #ifdef SHOW_POINT
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    #endif

    //获取大场景点云的小包围盒集合
    pcl::MomentOfInertiaEstimation<PointType> feature_extractor;
    feature_extractor.setInputCloud(cloud);
    feature_extractor.compute();
    Bbox_AABB aabb;
    feature_extractor.getAABB(aabb.min_point, aabb.max_point);
    std::cout << "min: " << aabb.min_point << std::endl;
    std::cout << "max: " << aabb.max_point << std::endl;
    std::vector<Bbox_AABB> bboxs;
    bboxs = CreatBboxs(aabb, stride_, kernel_);
    int batch_size = bboxs.size();
    std::cout << "batch size: " << batch_size << std::endl;

    // 拼接成tensor
    int sample_size = sample_point_pts.size();
    std::cout << "sample size: " << sample_size << std::endl;
    torch::Tensor com_pts_tensor = torch::ones({batch_size+sample_size, sample_num_, 3});
    TicToc get_pcs_time;
    for(int i = 0; i < bboxs.size(); i++){

        #ifdef SHOW_POINT_BOX
        // 显示Bbox
        std::cout << "---- " << i << "------" <<std::endl;
        std::cout << "min : " << bboxs[i].min_point << std::endl;
        std::cout << "max : " << bboxs[i].max_point << std::endl;
        viewer->addCube(bboxs[i].min_point.x,
                        bboxs[i].max_point.x,
                        bboxs[i].min_point.y,
                        bboxs[i].max_point.y,
                        bboxs[i].min_point.z,
                        bboxs[i].max_point.z,
                        0.1*i, 0.05*i, 0.0, to_string(i));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                            0.1, to_string(i));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
                                            4, to_string(i));
        #endif

        std::vector<int> indices;
        Eigen::Vector4f min_point(bboxs[i].min_point.x, bboxs[i].min_point.y, bboxs[i].min_point.z, 1);
        Eigen::Vector4f max_point(bboxs[i].max_point.x, bboxs[i].max_point.y, bboxs[i].max_point.z, 1);
        pcl::getPointsInBox(*cloud, min_point, max_point, indices);
        if(indices.size() == 0){
            continue;
        }

        pcl::PointCloud<PointType>::Ptr cloud_batch (new pcl::PointCloud<PointType>);
        pcl::copyPointCloud(*cloud, indices, *cloud_batch);
        torch::Tensor pts_batch_tensor = PcsToTensor(cloud_batch, sample_num_).to(torch::kFloat); // sample_num * 3
        com_pts_tensor.index({i, "..."}) = pts_batch_tensor;
    }
    get_pcs_time.toc(" get pointcloud batch: ");

    for(int j = 0; j < sample_size; j++){
        torch::Tensor pts_batch_tensor = PcsToTensor(sample_point_pts[j], sample_num_).to(torch::kFloat);
        com_pts_tensor.index({j+batch_size, "..."}) = pts_batch_tensor;
    }

    std::cout << " size com: " << com_pts_tensor.sizes() << std::endl;  // batch_size * sameple * 3
    // 注意网络的输入格式是 batch_size * channels * point_num
    torch::Tensor inputs = com_pts_tensor.transpose(2, 1).to(torch::kCUDA);
    std::cout << " size inputs: " <<inputs.sizes() << std::endl; 

    // 前向推理
    torch::jit::script::Module model;
    model = torch::jit::load(model_dir_);
    model.to(torch::kCUDA);
    model.eval();
    at::Tensor output = model.forward({inputs}).toTensor();

    auto pred = output.argmax(1).to(torch::kCPU);
    std::cout <<"pred: " << pred.sizes() << std::endl;

    // 生成返回点云
    pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_pred_temp (new pcl::PointCloud<pcl::PointXYZHSV> ());
    pcl::copyPointCloud(*cloud, *cloud_pred_temp);
    pcl::copyPointCloud(*cloud, *cloud_pred);  //cloud_pred是返回值

    for(int i=0; i<cloud_pred_temp->points.size(); i++){
        cloud_pred_temp->points[i].h = 0; //用来表征总共的预测次数
        cloud_pred_temp->points[i].s = 0; //用来表征被预测为不可行的次数
        cloud_pred->points[i].intensity = 0;
    }

    // 将预测结果赋值到点云强度上
    for(int i = 0; i < bboxs.size(); i++){
        std::vector<int> indices;
        Eigen::Vector4f min_point(bboxs[i].min_point.x, bboxs[i].min_point.y, bboxs[i].min_point.z, 1);
        Eigen::Vector4f max_point(bboxs[i].max_point.x, bboxs[i].max_point.y, bboxs[i].max_point.z, 1);
        pcl::getPointsInBox(*cloud_pred, min_point, max_point, indices);
        if(pred[i].item().toInt() == 1){
            for (int j = 0; j < indices.size(); j++) {
                cloud_pred_temp->points[indices[j]].h += 1;
                cloud_pred_temp->points[indices[j]].s += 1;
            }
        }

        if(pred[i].item().toInt() == 0){
            for (int j = 0; j < indices.size(); j++) {
                cloud_pred_temp->points[indices[j]].h += 1;
            }
        }
    }

    for (int i = 0; i < cloud_pred->points.size(); i++)
    {
        // std::cout << "s: " << cloud_pred_temp->points[i].s << " h " << cloud_pred_temp->points[i].h << std::endl;
        float pro = 10.0 * cloud_pred_temp->points[i].s / cloud_pred_temp->points[i].h; // 放缩到0-10 之间
        cloud_pred->points[i].intensity = pro;
        if (pro < 5)
        {
            cloud_pred_pass->points.push_back(cloud_pred->points[i]);
        }
        
    }

    //生成各个轨迹采样点的预测结果
    for (size_t i = 0; i < sample_size; i++)
    {
        sample_point_pred.push_back(pred[i+batch_size].item().toInt()); // sample_point_pred是返回值
    }
    std::cout << "sample pred size:" << sample_point_pred.size() << std::endl;
    


    }


/**
 * @brief PointPred 预测点云  --> 这里的输入是sample_trj采样到的点云
 * 　　　　　　　　　　　　将预测采样轨迹上的
 * @param sample_point_pts 【输入】 采样轨迹点所采样到的点云
 * @param sample_point_pred 【返回值】 采样轨迹点的预测结果
 */
void NetPred::PointPred(std::vector<pcl::PointCloud<PointType>::Ptr > sample_point_pts,
                        std::vector<int> &sample_point_pred){
                            
    // 拼接成tensor
    int sample_size = sample_point_pts.size();
    std::cout << "sample size: " << sample_size << std::endl;
    torch::Tensor com_pts_tensor = torch::ones({sample_size, sample_num_, 3});

    for(int j = 0; j < sample_size; j++){
        torch::Tensor pts_batch_tensor = PcsToTensor(sample_point_pts[j], sample_num_);
        com_pts_tensor.index({j, "..."}) = pts_batch_tensor;
    }

    std::cout << " size com: " << com_pts_tensor.sizes() << std::endl;  // batch_size * sameple * 3
    // 注意网络的输入格式是 batch_size * channels * point_num
    torch::Tensor inputs = com_pts_tensor.transpose(2, 1).to(torch::kCUDA);
    std::cout << " size inputs: " <<inputs.sizes() << std::endl; 

    // 前向推理
    // model_ = torch::jit::load(model_dir_);
    model_.to(torch::kCUDA);
    model_.eval();
    at::Tensor output = model_.forward({inputs}).toTensor();

    auto pred = output.argmax(1).to(torch::kCPU);
    std::cout <<"pred: " << pred.sizes() << std::endl;

    //生成各个轨迹采样点的预测结果
    for (size_t i = 0; i < sample_size; i++)
    {
        sample_point_pred.push_back(pred[i].item().toInt()); // sample_point_pred是返回值
    }
    
}

/**
 * @brief PointPred 预测点云
 *              这里将局部点云采用滑动窗口的方式分割为网络需要的点云大小(2M*2M)
 * 　　　　　　　　然后将预测的结果写入新点云的点云强度中, 所以强度越大, 危险性越高
 * @param cloud          输入点云指针, 这里的点云需要是以小车为中心的(小车前进路线上的)某块局部点云(10M*10M或其他大小)
 * @param cloud_pred　    带有预测结果的点云指针 [返回值]
 * @param cloud_pred_pass 预测结果为通过的点云指针 [返回值]
 */
void NetPred::PointPred(pcl::PointCloud<PointType>::Ptr cloud, pcl::PointCloud<PointType>::Ptr &cloud_pred, pcl::PointCloud<PointType>::Ptr &cloud_pred_pass){

    #ifdef SHOW_POINT
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    #endif

    std::cout << "=== [NetPred] ==> PointPred Start!!! "  << cloud->points.size() << std::endl;

    //获取大场景点云的小包围盒集合
    pcl::MomentOfInertiaEstimation<PointType> feature_extractor;
    feature_extractor.setInputCloud(cloud);
    feature_extractor.compute();
    Bbox_AABB aabb;
    feature_extractor.getAABB(aabb.min_point, aabb.max_point);
    std::cout << "min: " << aabb.min_point << std::endl;
    std::cout << "max: " << aabb.max_point << std::endl;
    std::vector<Bbox_AABB> bboxs;
    bboxs = CreatBboxs(aabb, stride_, kernel_);
    int batch_size = bboxs.size();
    std::cout << "size: " << bboxs.size() << std::endl;

    // 拼接成tensor
    torch::Tensor com_pts_tensor = torch::ones({batch_size, sample_num_, 3});
    for(int i = 0; i < bboxs.size(); i++){

        #ifdef SHOW_POINT_BOX
        // 显示Bbox
        std::cout << "---- " << i << "------" <<std::endl;
        std::cout << "min : " << bboxs[i].min_point << std::endl;
        std::cout << "max : " << bboxs[i].max_point << std::endl;
        viewer->addCube(bboxs[i].min_point.x,
                        bboxs[i].max_point.x,
                        bboxs[i].min_point.y,
                        bboxs[i].max_point.y,
                        bboxs[i].min_point.z,
                        bboxs[i].max_point.z,
                        0.1*i, 0.05*i, 0.0, to_string(i));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                            0.1, to_string(i));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
                                            4, to_string(i));
        #endif

        std::vector<int> indices;
        Eigen::Vector4f min_point(bboxs[i].min_point.x, bboxs[i].min_point.y, bboxs[i].min_point.z, 1);
        Eigen::Vector4f max_point(bboxs[i].max_point.x, bboxs[i].max_point.y, bboxs[i].max_point.z, 1);
        pcl::getPointsInBox(*cloud, min_point, max_point, indices);
        if(indices.size() == 0){
            continue;
        }

        pcl::PointCloud<PointType>::Ptr cloud_batch (new pcl::PointCloud<PointType>);
        pcl::copyPointCloud(*cloud, indices, *cloud_batch);
        torch::Tensor pts_batch_tensor = PcsToTensor(cloud_batch, sample_num_).to(torch::kFloat); // sample_num * 3
        com_pts_tensor.index({i, "..."}) = pts_batch_tensor;
    }

    std::cout << " size com: " << com_pts_tensor.sizes() << std::endl;  // batch_size * sameple * 3
    // 注意网络的输入格式是 batch_size * channels * point_num
    torch::Tensor inputs = com_pts_tensor.transpose(2, 1).to(torch::kCUDA);
    std::cout << " size inputs: " <<inputs.sizes() << std::endl; 

    // 前向推理
    model_ = torch::jit::load(model_dir_);
    model_.to(torch::kCUDA);
    model_.eval();
    at::Tensor output = model_.forward({inputs}).toTensor();
    // std::cout << output << std::endl; //如果预测失败可以查看output是否有0.0 或nan, 若有表明输入rensor中有inf值，需要处理

    auto pred = output.argmax(1).to(torch::kCPU);
    // std::cout <<"pred: " << pred.sizes() << std::endl;
    // std::cout <<"pred list: " << pred << std::endl;

    // 生成返回点云
    pcl::PointCloud<pcl::PointXYZHSV>::Ptr cloud_pred_temp (new pcl::PointCloud<pcl::PointXYZHSV> ());
    pcl::copyPointCloud(*cloud, *cloud_pred_temp);
    pcl::copyPointCloud(*cloud, *cloud_pred);

    for(int i=0; i<cloud_pred_temp->points.size(); i++){
        cloud_pred_temp->points[i].h = 0; //用来表征总共的预测次数
        cloud_pred_temp->points[i].s = 0; //用来表征被预测为不可行的次数
        cloud_pred->points[i].intensity = 0;
    }

    // 将预测结果赋值到点云强度上
    for(int i = 0; i < bboxs.size(); i++){
        std::vector<int> indices;
        Eigen::Vector4f min_point(bboxs[i].min_point.x, bboxs[i].min_point.y, bboxs[i].min_point.z, 1);
        Eigen::Vector4f max_point(bboxs[i].max_point.x, bboxs[i].max_point.y, bboxs[i].max_point.z, 1);
        pcl::getPointsInBox(*cloud_pred, min_point, max_point, indices);
        if(pred[i].item().toInt() == 1){
            for (int j = 0; j < indices.size(); j++) {
                cloud_pred_temp->points[indices[j]].h += 1;
                cloud_pred_temp->points[indices[j]].s += 1;
            }
        }

        if(pred[i].item().toInt() == 0){
            for (int j = 0; j < indices.size(); j++) {
                cloud_pred_temp->points[indices[j]].h += 1;
            }
        }
    }

    for (int i = 0; i < cloud_pred->points.size(); i++)
    {
        // std::cout << "s: " << cloud_pred_temp->points[i].s << " h " << cloud_pred_temp->points[i].h << std::endl;
        float pro = 10.0 * cloud_pred_temp->points[i].s / cloud_pred_temp->points[i].h; // 放缩到0-10 之间
        cloud_pred->points[i].intensity = pro;
        if (pro < 5)
        {
            cloud_pred_pass->points.push_back(cloud_pred->points[i]);
        }
        
    } 
    

    #ifdef SHOW_POINT
    pcl::visualization::PointCloudColorHandlerGenericField<PointType> fildColor(cloud_pred, "intensity"); // 按照 intensity 强度字段进行渲染
    viewer->setBackgroundColor(1, 1, 1);
    viewer->addPointCloud<PointType>(cloud_pred, fildColor, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3.0, "sample cloud"); // 设置点云大小
    viewer->addCoordinateSystem(1.0);

    while (!viewer->wasStopped()){
        viewer->spinOnce();
    }
    #endif

    // return cloud_pred, cloud_pred_pass;

}

/**
 * @brief CreatBboxs 根据大点云的包围盒，生成小点云的包围盒坐标集合
 * @param aabb     大点云的AABB包围盒
 * @param stride   小包围盒中心滑窗步长
 * @param kernel   小包围盒大小 长=宽=kernel 高=2*kernel+2
 * @return std::vector<Bbox_AABB>
 */
std::vector<Bbox_AABB> NetPred::CreatBboxs(Bbox_AABB aabb, float stride, float kernel){
    std::vector<Bbox_AABB> bboxs;
    int range_i = ceil((aabb.max_point.x - aabb.min_point.x - kernel)/stride);
    int range_j = ceil((aabb.max_point.y - aabb.min_point.y - kernel)/stride);
    std::cout << range_i << " " << range_j << std::endl;
    pcl::PointXYZ center;
    for(int j = 0; j < range_j+1; j++){
        for(int i = 0; i < range_i+1; i++){
            Bbox_AABB bbox;
            center.x = aabb.min_point.x + kernel/2.0 + stride*i;
            center.y = aabb.min_point.y + kernel/2.0 + stride*j;
            center.z = (aabb.min_point.z + aabb.max_point.z) / 2;
            bbox.min_point.x = center.x - kernel/2.0;
            bbox.min_point.y = center.y - kernel/2.0;
            bbox.min_point.z = center.z - kernel - 1;
            bbox.max_point.x = center.x + kernel/2.0;
            bbox.max_point.y = center.y + kernel/2.0;
            bbox.max_point.z = center.z + kernel + 1;
            bboxs.push_back(bbox);
        }
    }

    return bboxs;
}

/**
 * @brief PcsToTensor 将PCL中的pointcloud数据格式点云转换为torch中的tensor格式
 *        其中进行额外两步操作，  1. 将点云重采样到一定数量
 *                             2. 将点云进行坐标归一化
 * @param cloud          输入点云指针
 * @param sample_points  随机采样的点数
 * @return torch::tensor
 */
torch::Tensor NetPred::PcsToTensor(pcl::PointCloud<PointType>::Ptr cloud, int sample_points){
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
    //std::cout << " remove nan point Num: " << cloud->points.size() << std::endl;

    // 采样到固定数量
    int pts_num = cloud->points.size();
    pcl::PointCloud<PointType>::Ptr cloud_out (new pcl::PointCloud<PointType>);
    if(pts_num >= sample_points){   //如果点数足够多直接使用RandomSample方法
        pcl::RandomSample<PointType> rs;
        rs.setInputCloud(cloud);
        rs.setSample(sample_points);
        rs.filter(*cloud_out);
    } else{  // 如果点云点数少于采样点数
        pcl::PointCloud<PointType>::Ptr temp (new pcl::PointCloud<PointType>);
        pcl::copyPointCloud(*cloud, *temp);
        int a = sample_points / pts_num;
        while (a >= 0){
            *cloud += *temp;
            a--;
        }
        // std::cout << " add point Num: " << cloud->points.size() << std::endl;
        pcl::RandomSample<PointType> rs;
        rs.setInputCloud(cloud);
        rs.setSample(sample_points);
        rs.filter(*cloud_out);
        //std::cout << "num after sample: " << cloud_out->points.size() << std::endl;
    }

    // 归一化坐标
    float x_sum, y_sum, z_sum;
    x_sum = y_sum = z_sum = 0;
    for(auto a: cloud_out->points){
        x_sum += a.x;
        y_sum += a.y;
        z_sum += a.z;
    }
    float x_mean, y_mean, z_mean;
    x_mean = x_sum / cloud_out->points.size();
    y_mean = y_sum / cloud_out->points.size();
    z_mean = z_sum / cloud_out->points.size();

    //std::cout << "center: " << x_mean << " " << y_mean << " " << z_mean << std::endl;
    // PCL函数计算质心
    //Eigen::Vector4f centroid;					// 质心
    //pcl::compute3DCentroid(*cloud, centroid);	// 齐次坐标，（c0,c1,c2,1）
    //std::cout << "center:   --" << centroid.transpose() << std::endl;

    std::vector<float> pcs_1d;
    for (int i = 0; i < cloud_out->points.size(); i++) {
        cloud_out->points[i].x = cloud_out->points[i].x - x_mean;
        cloud_out->points[i].y = cloud_out->points[i].y - y_mean;
        cloud_out->points[i].z = cloud_out->points[i].z - z_mean;
        pcs_1d.push_back(cloud_out->points[i].x);
        pcs_1d.push_back(cloud_out->points[i].y);
        pcs_1d.push_back(cloud_out->points[i].z);
    }

    
    //std::cout << " +++++++ " << std::endl;
    //for(auto a: cloud_out->points){
    //    std::cout << a << std::endl;
    //}
    

    torch::Tensor pcs_tensor = torch::from_blob(pcs_1d.data(), {sample_points, 3}, torch::kFloat);
    //std::cout << "---\n" << pcs_tensor << std::endl;
    //std::cout << pcs_tensor.sizes() <<std::endl;
    // 由于pcs_1d析构后会导致tensor中的值变化
    // https://www.cxybb.com/article/weixin_45415546/99639632
    return pcs_tensor.clone();
}
