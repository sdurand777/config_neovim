local ls = require("luasnip") --{{{
local s = ls.s --> snippet
local i = ls.i --> insert node
local t = ls.t --> text node

local d = ls.dynamic_node
local c = ls.choice_node
local f = ls.function_node
local sn = ls.snippet_node

local fmt = require("luasnip.extras.fmt").fmt
local rep = require("luasnip.extras").rep

local snippets, autosnippets = {}, {} --}}}

local group = vim.api.nvim_create_augroup("type de fichier cpp ", { clear = true })
local file_pattern = "*.cpp"

local function cs(trigger, nodes, opts) --{{{
    local snippet = s(trigger, nodes)
    local target_table = snippets

    local pattern = file_pattern
    local keymaps = {}

    if opts ~= nil then
        -- check for custom pattern
        if opts.pattern then
            pattern = opts.pattern
        end

        -- if opts is a string
        if type(opts) == "string" then
            if opts == "auto" then
                target_table = autosnippets
            else
                table.insert(keymaps, { "i", opts })
            end
        end

        -- if opts is a table
        if opts ~= nil and type(opts) == "table" then
            for _, keymap in ipairs(opts) do
                if type(keymap) == "string" then
                    table.insert(keymaps, { "i", keymap })
                else
                    table.insert(keymaps, keymap)
                end
            end
        end

        -- set autocmd for each keymap
        if opts ~= "auto" then
            for _, keymap in ipairs(keymaps) do
                vim.api.nvim_create_autocmd("BufEnter", {
                    pattern = pattern,
                    group = group,
                    callback = function()
                        vim.keymap.set(keymap[1], keymap[2], function()
                            ls.snip_expand(snippet)
                        end, { noremap = true, silent = true, buffer = true })
                    end,
                })
            end
        end
    end

    table.insert(target_table, snippet) -- insert snippet into appropriate table
end --}}}


-- Ecrire ses snippets lua on peut utiliser le snipet luasnippet 
cs("eigen", fmt( -- set eigen
[[using namespace Eigen;]], {}))


cs("opencv", fmt( -- set opencv 
[[using namespace cv]], {}))


cs("std", fmt( -- using namespace std 
[[using namespace std]], {}))


cs("dvar", fmt( -- display a value
[[
std::cout << "{} : " << {} << std::endl; 
]], {
  i(1, ""),
  rep(1)
  }))


 cs("network_client_example", fmt( -- network client example cpp
 [[
 #include <iostream>
 #include <boost/asio.hpp>
 #include <opencv2/opencv.hpp>
 using namespace boost::asio;
 using namespace cv;
 
 int main(int argc, char** argv) {{
     if (argc != 3) {{
         std::cerr << "Usage: client <host> <port>\n";
         return 1;
     }}
 
     io_service service;
     ip::tcp::socket socket(service);
     ip::tcp::resolver resolver(service);
     boost::asio::connect(socket, resolver.resolve({{ argv[1], argv[2] }}));
 
     namedWindow("Client window", WINDOW_NORMAL);
 
     while (true) {{
         // Receive image size
         uint32_t size;
         boost::asio::read(socket, boost::asio::buffer(&size, sizeof(size)));
 
         // Receive image data
         std::vector<uchar> buffer(size);
         boost::asio::read(socket, boost::asio::buffer(buffer));
 
         // Decode image
         cv::Mat img = cv::imdecode(buffer, cv::IMREAD_COLOR);
 
         // Show image
         imshow("Client window", img);
         waitKey(1);
     }}
 
     return 0;
 }}
 ]]
 , {}))
 
 
 
 
 cs("network_server_example", fmt( -- network server example 
 [[
 #include <iostream>
 #include <opencv2/opencv.hpp>
 #include <boost/asio.hpp>
 #include <boost/thread.hpp>
 #include <cstdint>
 
 using boost::asio::ip::tcp;
 
 int main(int argc, char* argv[])
 {{
     try {{
         // Initialisation de la webcam
         cv::VideoCapture cap(0);
         if (!cap.isOpened()) {{
             std::cerr << "Erreur d'ouverture de la webcam\n";
             return 1;
         }}
 
         // Initialisation du serveur
         boost::asio::io_service io_service;
         tcp::acceptor acceptor(io_service, tcp::endpoint(tcp::v4(), 1234));
 
         std::cout << "Serveur en attente de connexion sur le port 1234\n";
 
         // Attendre une connexion
         tcp::socket socket(io_service);
         acceptor.accept(socket);
 
         std::cout << "Client connecté\n";
 
         // Envoyer les images de la webcam
         cv::Mat frame;
         while (cap.read(frame)) {{
             // Convertir l'image en tableau de bytes
             std::vector<uchar> buf;
             cv::imencode(".jpg", frame, buf);
 
             // Envoyer la taille de l'image
             uint32_t size = buf.size();
             boost::asio::write(socket, boost::asio::buffer(&size, sizeof(uint32_t)));
 
             // Envoyer l'image
             boost::asio::write(socket, boost::asio::buffer(buf));
 
             // Attendre un peu pour éviter d'envoyer trop rapidement
             boost::this_thread::sleep(boost::posix_time::milliseconds(50));
         }}
 
         // Envoyer un message de fin de connexion au client
         uint32_t zero = 0;
         boost::asio::write(socket, boost::asio::buffer(&zero, sizeof(uint32_t)));
 
         std::cout << "Toutes les images ont été envoyées\n";
     }}
     catch (std::exception& e) {{
         std::cerr << e.what() << std::endl;
     }}
 
     return 0;
 }}
 ]]
 , {
   }))
 
 
 
 cs("network_server_with_thread_example", fmt( -- network server with thread example
 [[
 #include <iostream>
 #include <opencv2/opencv.hpp>
 #include <boost/asio.hpp>
 #include <boost/thread.hpp>
 #include <cstdint>
 #include <mutex>
 #include <condition_variable>
 #include <thread>
 
 using boost::asio::ip::tcp;
 
 std::mutex mtx;
 cv::Mat current_frame;
 std::condition_variable cvt;
 bool is_running = true;
 
 void display_thread() {{
     cv::namedWindow("Webcam", cv::WINDOW_NORMAL);
     cv::resizeWindow("Webcam", 640, 480);
 
     while (is_running) {{
         std::unique_lock<std::mutex> lock(mtx);
         cvt.wait(lock);
         cv::imshow("Webcam", current_frame);
         cvt.notify_all();
         cv::waitKey(1);
     }}
 }}
 
 void send_image(tcp::socket& socket) {{
     while (is_running) {{
         std::unique_lock<std::mutex> lock(mtx);
         cvt.wait(lock);
         if (!current_frame.empty()) {{
             std::vector<uchar> buf;
             cv::imencode(".jpg", current_frame, buf);
             uint32_t size = buf.size();
             boost::asio::write(socket, boost::asio::buffer(&size, sizeof(uint32_t)));
             boost::asio::write(socket, boost::asio::buffer(buf));
         }}
         cvt.notify_all();
         std::this_thread::sleep_for(std::chrono::milliseconds(50));
     }}
 }}
 
 void server_thread() {{
     try {{
         boost::asio::io_service io_service;
         tcp::acceptor acceptor(io_service, tcp::endpoint(tcp::v4(), 1234));
 
         std::cout << "Serveur en attente de connexion sur le port 1234\n";
 
         while (is_running) {{
             tcp::socket socket(io_service);
             acceptor.accept(socket);
 
             std::cout << "Client connecté\n";
 
             boost::thread send([&socket]() {{
                 send_image(socket);
             }});
 
             send.join();
         }}
     }}
     catch (std::exception& e) {{
         std::cerr << e.what() << std::endl;
     }}
 }}
 
 int main(int argc, char* argv[]) {{
     boost::thread display(display_thread);
     boost::thread server(server_thread);
 
     cv::VideoCapture cap(0);
     if (!cap.isOpened()) {{
         std::cerr << "Erreur d'ouverture de la webcam\n";
         return 1;
     }}
 
     cv::Mat frame;
     while (is_running) {{
         if (!cap.read(frame)) {{
             std::cerr << "Erreur de lecture de l'image de la webcam\n";
             break;
         }}
         std::unique_lock<std::mutex> lock(mtx);
         current_frame = frame.clone();
         cvt.notify_all();
         cvt.wait(lock);
         std::this_thread::sleep_for(std::chrono::milliseconds(1));
     }}
 
     is_running = false;
     display.join();
     server.join();
 
     return 0;
 }}
 ]]
 , {
   }))
 
 
 
 cs("Eigen_packages", fmt( -- eigen packages 
 [[
 #include <Eigen/Core>
 #include <Eigen/Dense>
 using namespace Eigen;
 ]], {
   }))
 
 
 cs("Eigen_direct_inverse", fmt( -- eigen compute direct inverse 
 [[
 // compute direct inverse Ax = b solve x
 Matrix<double, A.rows(), 1> x = A.inverse() * b;
 cout << "time of normal inverse is "
    << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
 cout << "x = " << x.transpose() << endl;
 time_stt = clock();
 ]], {
   }))
 
 
 cs("Eigen_QR_decomposition", fmt( -- eigen compute QR decomposition 
 [[
 // compute matrix decomposition
 x = A.colPivHouseholderQr().solve(b);
 cout << "time of Qr decomposition is "
    << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
 cout << "x = " << x.transpose() << endl;
 time_stt = clock();
 ]], {
   }))
 
 
 cs("Eigen_cholesky_decomposition", fmt( -- eigen compute cholesky decomposition 
 [[
 // cholesky decomposition to solve equation
 x = A.ldlt().solve(b);
 cout << "time of ldlt decomposition is "
    << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
 cout << "x = " << x.transpose() << endl;
 ]], {
   }))
 
 
 
 
 cs("Eigen_rotation_vector", fmt( -- eigen rotation vectoc
 [[
 AngleAxisd rotation_vector(M_PI / 4, Vector3d(0, 0, 1));     // 45 degrees along Z
 ]], {
   }))
 
 
 cs("Eigen_transform_matrix", fmt( -- eigen transfo matrix 
 [[
 // Eigen::Isometry
 Isometry3d T = Isometry3d::Identity();                // although 3D 4x4 matrix
 AngleAxisd rotation_vector(M_PI / 4, Vector3d(0, 0, 1));     // 45 degrees along Z
 T.rotate(rotation_vector);                                     // apply rotation
 T.pretranslate(Vector3d(1, 3, 4));                     // apply translation
 cout << "Transform matrix = \n" << T.matrix() << endl;
 ]], {
   }))
 
 cs("Eigen_quaternion_from_rotation_vector", fmt( -- eigen compute quaternion from rotatoin vector
 [[
 Quaterniond q = Quaterniond(rotation_vector);
 cout << "quaternion from rotation vector = " << q.coeffs().transpose()
    << endl;   // the first three are imaginary parts and the first is the real part
 ]], {
   }))
 
 
 cs("Eigen_coord_transfo", fmt( -- eigen example of coordinate transfo 
 [[
 // quaternion R1 R2
 Quaterniond q1(0.35, 0.2, 0.3, 0.1), q2(-0.5, 0.4, -0.1, 0.2);
 // normalize quaternion before use
 q1.normalize();
 q2.normalize();
 Vector3d t1(0.3, 0.1, 0.1), t2(-0.1, 0.5, 0.3);
 // point in R1 to compute in R2
 Vector3d p1(0.5, 0, 0.2);
 Isometry3d T1w(q1), T2w(q2);
 T1w.pretranslate(t1);
 T2w.pretranslate(t2);
 Vector3d p2 = T2w * T1w.inverse() * p1;
 cout << endl << p2.transpose() << endl;
 ]], {
   }))
 
 
 cs("Pangolin_display_pose", fmt( -- Pangolin example to display poses
 [[
 void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses) {{
   // create pangolin window and plot the trajectory
   pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
   glEnable(GL_DEPTH_TEST);
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   // set up cam view
   pangolin::OpenGlRenderState s_cam(
     pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
     pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
   );
   // set up cam
   pangolin::View &d_cam = pangolin::CreateDisplay()
     .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
     .SetHandler(new pangolin::Handler3D(s_cam));
   while (pangolin::ShouldQuit() == false) {{
     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
     d_cam.Activate(s_cam);
     glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
     glLineWidth(2);
     for (size_t i = 0; i < poses.size(); i++) {{
       // draw the three axes for each poses to show each frame
       if (i == 0)
             cout << "pose : " << endl << poses[i].matrix() << endl;
       Vector3d Ow = poses[i].translation();
       // overloaded multiplication with poses transform
       Vector3d Xw = poses[i] * (0.1 * Vector3d(1, 0, 0));
       Vector3d Yw = poses[i] * (0.1 * Vector3d(0, 1, 0));
       Vector3d Zw = poses[i] * (0.1 * Vector3d(0, 0, 1));
       // define the three lines
       glBegin(GL_LINES);
       glColor3f(1.0, 0.0, 0.0);
       glVertex3d(Ow[0], Ow[1], Ow[2]);
       glVertex3d(Xw[0], Xw[1], Xw[2]);
       glColor3f(0.0, 1.0, 0.0);
       glVertex3d(Ow[0], Ow[1], Ow[2]);
       glVertex3d(Yw[0], Yw[1], Yw[2]);
       glColor3f(0.0, 0.0, 1.0);
       glVertex3d(Ow[0], Ow[1], Ow[2]);
       glVertex3d(Zw[0], Zw[1], Zw[2]);
       glEnd();
     }}
     // draw connection between poses
     for (size_t i = 0; i < poses.size(); i++) {{
       glColor3f(0.0, 0.0, 0.0);
       glBegin(GL_LINES);
       auto p1 = poses[i], p2 = poses[i + 1];
       glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
       glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
       glEnd();
     }}
     pangolin::FinishFrame();
     usleep(5000);   // sleep 5 ms
   }}
 }}
 ]], {
   }))
 
 
 
 cs("Pangolin_package", fmt( -- Pangolin package 
 [[
 #include <pangolin/pangolin.h>
 ]], {
   }))
 
 
 cs("Pangolin_cmake", fmt( -- Pangolin cmake 
 [[
 find_package(Pangolin REQUIRED)
 include_directories(${{Pangolin_INCLUDE_DIRS}})
 add_executable(main main.cpp)
 target_link_libraries(main ${{Pangolin_LIBRARIES}})
 ]], {
   }))
 
 
 
 cs("Sophus_package", fmt( -- Sophus package 
 [[
 #include "sophus/se3.hpp"
 ]], {
   }))
 
 
 
 cs("Sophus_cmake", fmt( -- Sophus cmake 
 [[
 find_package(Sophus REQUIRED)
 include_directories("/usr/include/eigen3")
 add_executable(main main.cpp)
 target_link_libraries(main Sophus::Sophus)
 ]], {
   }))
 
 
 cs("Sophus_SO3_from_rot_or_quat", fmt( -- sophus exemple conversion SO3 from rot or quat 
 [[
 // matrix de rotation on utilise un vecteur rotation transforme en matrice de rotation
 Matrix3d R = AngleAxisd(M_PI / 2, Vector3d(0, 0, 1)).toRotationMatrix();
 // definition du quaternion
 Quaterniond q(R);
 Sophus::SO3d SO3_R(R);
 Sophus::SO3d SO3_q(q);
 ]], {
   }))
 
 
 cs("Sophus_SO3_lie_algebra_manipulatoins", fmt( -- sophus exaaple so3 lie algebra manips 
 [[
 // log map lie algebra
 Vector3d so3 = SO3_R.log();
 cout << "so3 = " << so3.transpose() << endl;
 // hat get the skew symetric matrix
 cout << "so3 hat=\n" << Sophus::SO3d::hat(so3) << endl;
 // conversion back to vector
 cout << "so3 hat vee= " << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose() << endl;
 // update manipulation so3
 Vector3d update_so3(1e-4, 0, 0);
 Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
 cout << "SO3 updated = \n" << SO3_updated.matrix() << endl;
 ]], {
   }))
 
 
 
 cs("Sophus_SE3_conversion", fmt( -- sophus SE3 conversion from eigen 
 [[
 // similar se3
 Vector3d t(1, 0, 0);           // translation
 Sophus::SE3d SE3_Rt(R, t);           // R, t
 Sophus::SE3d SE3_qt(q, t);            // q, t
 cout << "SE3 from R,t= \n" << SE3_Rt.matrix() << endl;
 cout << "SE3 from q,t= \n" << SE3_qt.matrix() << endl;
 cout << "should be equal" << endl;
 ]], {
   }))
 
 
 cs("Sophus_se3_lie_algebra_manips", fmt( -- sophus lie algebra manips 
 [[
 // lie algebra is 6d vector
 typedef Eigen::Matrix<double, 6, 1> Vector6d;
 Vector6d se3 = SE3_Rt.log();
 cout << "se3 = " << se3.transpose() << endl;
 // check hat and hat vee conversion
 cout << "se3 hat = \n" << Sophus::SE3d::hat(se3) << endl;
 cout << "se3 hat vee = " << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose() << endl;
 // check update
 Vector6d update_se3;
 update_se3.setZero();
 update_se3(0, 0) = 1e-4;
 Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
 cout << "SE3 updated = " << endl << SE3_updated.matrix() << endl;
 ]], {
   }))
 
 
 
 cs("OpenCV_cmake", fmt( -- opencv cmake 
 [[
 find_package(OpenCV REQUIRED)
 include_directories(${{OpenCV_INCLUDE_DIRS}})
 add_executable(main main.cpp)
 target_link_libraries(main ${{OpenCV_LIBS}})
 ]], {
   }))
 
 
 
 cs("OpenCV_package", fmt( -- opencv include package 
 [[
 #include <opencv2/opencv.hpp>
 ]], {
   }))
 
 
 
 cs("OpenCV_check_error_open", fmt( -- opencv check error open 
 [[
 // check error to open image
 if (image.data == nullptr) {{ 
 cerr << "error opening no data" << endl;
 return 0;
 }}
 ]], {
   }))
 
 
 
 
 cs("OpenCV_check_error_channels", fmt( -- opencv check error channels 
 [[
 // check image type C1 or C3 else error
 if (image.type() != CV_8UC1 && image.type() != CV_8UC3) {{
     cout << "error with image channels" << endl;
     return 0;
 }}
 ]], {
   }))
 
 
 cs("OpenCV_go_through_umage", fmt( -- opencv go through image 
 [[
 // timer to go through image
 chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
 for (size_t y = 0; y < image.rows; y++) {{
     // cv::Mat::ptr
     unsigned char *row_ptr = image.ptr<unsigned char>(y);  // row_ptr
     for (size_t x = 0; x < image.cols; x++) {{
         unsigned char *data_ptr = &row_ptr[x * image.channels()]; // data_ptr
         for (int c = 0; c != image.channels(); c++) {{
             unsigned char data = data_ptr[c]; // data(x,y)
         }}
     }}
 }}
 chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
 chrono::duration<double> time_used = chrono::duration_cast < chrono::duration < double >> (t2 - t1);
 cout << "time：" << time_used.count() << "ms。" << endl;
 ]], {
   }))
 
 cs("OpenCV_manip_pixels_image", fmt( -- opencv manip image pixels 
 [[
 cv::Mat image_another = image;
 // put a square to black
 image_another(cv::Rect(0, 0, 100, 100)).setTo(0); 
 cv::imshow("image", image);
 cv::waitKey(0);
 ]], {
   }))
 
 
 cs("OpenCV_undistort_example", fmt( -- opencv undistort example 
 [[
 string image_file = "../distorted.png";
 
 int main(int argc, char **argv) {{
     // distortion coeffs
     double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
     // intrinsics parameters
     double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;
 
     cv::Mat image = cv::imread(image_file, 0);   // read image
     int rows = image.rows, cols = image.cols;
     cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1);   // define output image undistorted
 
     // undistort image
     for (int v = 0; v < rows; v++) {{
         for (int u = 0; u < cols; u++) {{
             double x = (u - cx) / fx, y = (v - cy) / fy;
             double r = sqrt(x * x + y * y);
             double x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
             double y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
             double u_distorted = fx * x_distorted + cx;
             double v_distorted = fy * y_distorted + cy;
 
             if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows) {{
                 image_undistort.at<uchar>(v, u) = image.at<uchar>((int) v_distorted, (int) u_distorted);
             }} else {{
                 image_undistort.at<uchar>(v, u) = 0;
             }}
         }}
     }}
 
     cv::imshow("distorted", image);
     cv::imshow("undistorted", image_undistort);
     cv::waitKey();
     return 0;
 }}
 ]], {
   }))
 
 
 cs("Pangolin_show_pointcloud", fmt( -- pangolin show pointcloud example 
 [[
 void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud) {{
 
     if (pointcloud.empty()) {{
         cerr << "Point cloud is empty!" << endl;
         return;
     }}
 
     pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
     glEnable(GL_DEPTH_TEST);
     glEnable(GL_BLEND);
     glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
 
     pangolin::OpenGlRenderState s_cam(
         pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
         pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
     );
 
     pangolin::View &d_cam = pangolin::CreateDisplay()
         .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
         .SetHandler(new pangolin::Handler3D(s_cam));
 
     while (pangolin::ShouldQuit() == false) {{
         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
 
         d_cam.Activate(s_cam);
         glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
 
         glPointSize(2);
         glBegin(GL_POINTS);
         for (auto &p: pointcloud) {{
             glColor3f(p[3], p[3], p[3]);
             glVertex3d(p[0], p[1], p[2]);
         }}
         glEnd();
         pangolin::FinishFrame();
         usleep(5000);   // sleep 5 ms
     }}
     return;
 }}
 ]], {
   }))
 
 
 
 cs("Opencv_stereo_disparity", fmt( -- opencv stereo disparity 
 [[
 // images
 string left_file = "../left.png";
 string right_file = "../right.png";
 
 // main app
 int main(int argc, char **argv) {{
 
     // intrinsics
     double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
     // baseline
     double b = 0.573;
     // read images
     cv::Mat left = cv::imread(left_file, 0);
     cv::Mat right = cv::imread(right_file, 0);
 
     // SGBM algo to compute disparity
     cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
         0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);    // SGBM algo
     cv::Mat disparity_sgbm, disparity;
     sgbm->compute(left, right, disparity_sgbm);
     disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);
 
     // point cloud
     vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;
 
     // compute depth
     for (int v = 0; v < left.rows; v++)
         for (int u = 0; u < left.cols; u++) {{
             if (disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0) continue;
 
             Vector4d point(0, 0, 0, left.at<uchar>(v, u) / 255.0);
         
             // conversion from pixel to points
             double x = (u - cx) / fx;
             double y = (v - cy) / fy;
             // compute depth at pixel position
             double depth = fx * b / (disparity.at<float>(v, u));
             // apply depth to compute 3d points
             point[0] = x * depth;
             point[1] = y * depth;
             point[2] = depth;
 
             pointcloud.push_back(point);
         }}
 
     cv::imshow("disparity", disparity / 96.0);
     cv::waitKey(0);
     showPointCloud(pointcloud);
     return 0;
 }}
 ]], {
   }))
 
 
 
 cs("Eigen_Gauss_Newton", fmt( -- Eigen Gauss Newton Algo
 [[
 int main(int argc, char **argv) {{
   double ar = 1.0, br = 2.0, cr = 1.0;         // ground truth values
   double ae = 2.0, be = -1.0, ce = 5.0;        // initial estimate
   int N = 100;                                 // Number of data points
   double w_sigma = 1.0;                        // sigma of the noise
   double inv_sigma = 1.0 / w_sigma;
   cv::RNG rng;                                 // random number generator
 
   vector<double> x_data, y_data;
   for (int i = 0; i < N; i++) {{
     double x = i / 100.0;
     x_data.push_back(x);
     y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
   }}
 
   //Gauss-Newton
   int iterations = 100; 
   double cost = 0, lastCost = 0;
 
   chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
   for (int iter = 0; iter < iterations; iter++) {{
 
     Matrix3d H = Matrix3d::Zero();             // Hessian = J^T W^{{-1}} J in Gauss-Newton
     Vector3d b = Vector3d::Zero();             // bias
     cost = 0;
 
     for (int i = 0; i < N; i++) {{
       double xi = x_data[i], yi = y_data[i];
       double error = yi - exp(ae * xi * xi + be * xi + ce);
       // compute jacobien
       Vector3d J;
       J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);  // de/da
       J[1] = -xi * exp(ae * xi * xi + be * xi + ce);  // de/db
       J[2] = -exp(ae * xi * xi + be * xi + ce);  // de/dc
 
       H += inv_sigma * inv_sigma * J * J.transpose();
       b += -inv_sigma * inv_sigma * error * J;
 
       cost += error * error;
     }}
 
     // Solve Hx=b
     Vector3d dx = H.ldlt().solve(b);
     if (isnan(dx[0])) {{
       cout << "result is nan!" << endl;
       break;
     }}
 
     if (iter > 0 && cost >= lastCost) {{
       cout << "cost: " << cost << ">= last cost: " << lastCost << ", break." << endl;
       break;
     }}
 
     ae += dx[0];
     be += dx[1];
     ce += dx[2];
 
     lastCost = cost;
 
     cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() <<
          "\t\testimated params: " << ae << "," << be << "," << ce << endl;
   }}
 
   chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
   chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
   cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
 
   cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
   return 0;
 }}
 ]], {
   }))
 
 
 
 
 cs("Ceres_cmake", fmt( -- ceres cmake include 
 [[
 # Ceres
 find_package(Ceres REQUIRED)
 include_directories(${{CERES_INCLUDE_DIRS}})
 # Eigen
 include_directories("/usr/include/eigen3")
 add_executable(main main.cpp)
 target_link_libraries(main ${{CERES_LIBRARIES}})
 ]], {}))
 
 
 cs("G2O_cmake", fmt( -- G2O cmake include 
 [[
 # g2o
 find_package(G2O REQUIRED)
 include_directories(${{G2O_INCLUDE_DIRS}})
 # Eigen
 include_directories("/usr/include/eigen3")
 add_executable(main main.cpp)
 target_link_libraries(main ${{G2O_CORE_LIBRARY}} ${{G2O_STUFF_LIBRARY}})
 ]], {
   }))
 
 
 
 cs("Ceres_package", fmt( -- ceres include package 
 [[
 #include <ceres/ceres.h>
 ]], {
   }))


 cs("Ceres_example", fmt( -- ceres simple example
 [[
 // struct to define error function used by autodiff in ceres to compute J
 struct CURVE_FITTING_COST {{
   CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {{}}
 
   // implement operator to compute the error
   template<typename T>
   bool operator()(
     const T *const abc, // the estimated variables 3d vector a b c
     T *residual) const {{
     residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]); // y-exp(ax^2+bx+c)
     return true;
   }}
 
   const double _x, _y;    // x,y members
 }};
 
 int main(int argc, char **argv) {{
   double ar = 1.0, br = 2.0, cr = 1.0;         // ground truth
   double ae = 2.0, be = -1.0, ce = 5.0;        // estimate initial
   int N = 100;                                 // number of data points
   double w_sigma = 1.0;                        // uncertainty of the noise
   double inv_sigma = 1.0 / w_sigma;
   cv::RNG rng;                                 // random generator
 
   vector<double> x_data, y_data;      // generate data
   for (int i = 0; i < N; i++) {{
     double x = i / 100.0;
     x_data.push_back(x);
     y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
   }}
 
   double abc[3] = {{ae, be, ce}};
 
   // problem definition
   ceres::Problem problem;
   for (int i = 0; i < N; i++) {{
     problem.AddResidualBlock(     // add residual block to the probleme 
                                   // residual type, output dimension, input dimension
       new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
         new CURVE_FITTING_COST(x_data[i], y_data[i])
       ),
       nullptr,            // kernel funciton
       abc                 // estimated variables
     );
   }}
 
   // deifne solver
   ceres::Solver::Options options;     // options
   options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  // cholesky decomposition
   options.minimizer_progress_to_stdout = true;   // print output
 
   ceres::Solver::Summary summary;
   chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
   ceres::Solve(options, &problem, &summary);  // do optimization
   chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
   chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
   cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
 
   // output log
   cout << summary.BriefReport() << endl;
   cout << "estimated a,b,c = ";
   for (auto a:abc) 
       cout << a << " ";
   cout << endl;
 
   return 0;
 }}
 ]], {
   }))



 cs("G2O_packages", fmt( -- g2o include packages 
 [[
 #include <g2o/core/g2o_core_api.h>
 #include <g2o/core/base_vertex.h>
 #include <g2o/core/base_unary_edge.h>
 #include <g2o/core/block_solver.h>
 #include <g2o/core/optimization_algorithm_levenberg.h>
 #include <g2o/core/optimization_algorithm_gauss_newton.h>
 #include <g2o/core/optimization_algorithm_dogleg.h>
 #include <g2o/solvers/dense/linear_solver_dense.h>
 ]], {
   }))



 cs("G2O_example", fmt( -- g2o example
 [[
 // vertex 3D vector a,b,c
 class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {{
 public:
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 
   // override the reset function
   virtual void setToOriginImpl() override {{
     _estimate << 0, 0, 0;
   }}
 
   // override the plus operator vector addition
   virtual void oplusImpl(const double *update) override {{
     _estimate += Eigen::Vector3d(update);
   }}
 
   // dummy read and write functions
   virtual bool read(istream &in) {{}}
   virtual bool write(ostream &out) const {{}}
 }};
 
 // edge 1D error term connected to only one vertex
 class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {{
 public:
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 
   CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {{}}
 
   // define the error term computation
   virtual void computeError() override {{
     const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);
     // estimate abc
     const Eigen::Vector3d abc = v->estimate();
     // compute error
     _error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
   }}
 
   // the jacobian we compute it ourselve
   virtual void linearizeOplus() override {{
     const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);
     const Eigen::Vector3d abc = v->estimate();
     double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
     _jacobianOplusXi[0] = -_x * _x * y;
     _jacobianOplusXi[1] = -_x * y;
     _jacobianOplusXi[2] = -y;
   }}
 
   // dummy read and write functions
   virtual bool read(istream &in) {{}}
   virtual bool write(ostream &out) const {{}}
 
 public:
   double _x;  // x see construction y = _measurement given
 }};
 
 
 // main app
 int main(int argc, char **argv) {{
   double ar = 1.0, br = 2.0, cr = 1.0;         // ground truth
   double ae = 2.0, be = -1.0, ce = 5.0;        // estimated initial values
   int N = 100;                                 // number of data points
   double w_sigma = 1.0;                        // noise sigma
   double inv_sigma = 1.0 / w_sigma;
   cv::RNG rng;                                 // random generator
 
   vector<double> x_data, y_data;      // generate data points
   for (int i = 0; i < N; i++) {{
     double x = i / 100.0;
     x_data.push_back(x);
     y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
   }}
 
   // typdef for convenience
   typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;  // block solver
   typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // linear solver
 
   // solver definition Gauss Newton algo
   auto solver = new g2o::OptimizationAlgorithmGaussNewton(
     std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
 
   g2o::SparseOptimizer optimizer;     // graph optimizer
   optimizer.setAlgorithm(solver);   // set algo
   optimizer.setVerbose(true);       // cout
 
   // add vertex
   CurveFittingVertex *v = new CurveFittingVertex();
   v->setEstimate(Eigen::Vector3d(ae, be, ce));
   v->setId(0);
   optimizer.addVertex(v);
 
   // add edges
   for (int i = 0; i < N; i++) {{
     CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
     edge->setId(i);
     edge->setVertex(0, v);                // connect to the vertex
     edge->setMeasurement(y_data[i]);      // measurement
     edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma)); // set information matrix
     optimizer.addEdge(edge);
   }}
 
   // do optimization
   cout << "start optimization" << endl;
   chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
   optimizer.initializeOptimization();
   optimizer.optimize(10);
   chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
   chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
   cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
 
   // get abc estimate
   Eigen::Vector3d abc_estimate = v->estimate();
   cout << "estimated model: " << abc_estimate.transpose() << endl;
 
   return 0;
 }}
 ]], {
   }))




cs("OpenCV_orb_features_example", fmt( -- orb example with orb features
[[
// main app
int main(int argc, char **argv) {{
  if (argc != 3) {{
    cout << "usage: feature_extraction img1 img2" << endl;
    return 1;
  }}

  // read images
  Mat img_1 = imread(argv[1]);
  Mat img_2 = imread(argv[2]);
  assert(img_1.data != nullptr && img_2.data != nullptr);

  //-- define detector extracto and matcher
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create(); 
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming"); // brute force hamming to match

  // Get features on both images
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  // Get descriptors on both images
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

  // display keypoints
  Mat outimg1;
  drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  imshow("ORB features", outimg1);

  //-- compute matches
  vector<DMatch> matches;
  t1 = chrono::steady_clock::now();
  matcher->match(descriptors_1, descriptors_2, matches);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;

  // remove bad matches
  auto min_max = minmax_element(matches.begin(), matches.end(),
                                [](const DMatch &m1, const DMatch &m2) {{ return m1.distance < m2.distance; }});

  // min max distances
  double min_dist = min_max.first->distance;
  double max_dist = min_max.second->distance;

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  // gather good matches empirical threshold 2*min_dist
  std::vector<DMatch> good_matches;
  for (int i = 0; i < descriptors_1.rows; i++) {{
    if (matches[i].distance <= max(2 * min_dist, 30.0)) {{
      good_matches.push_back(matches[i]);
    }}
  }}

  //-- display results
  Mat img_match;
  Mat img_goodmatch;
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
  imshow("all matches", img_match);
  imshow("good matches", img_goodmatch);
  waitKey(0);

  return 0;
}}
]], {
  }))



cs("OpenCV_features_package", fmt( -- opencv include packages for orb 
[[
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
]], {
  }))



cs("OpenCV_epipolar_package", fmt( -- opencv package to run epipolar example with openv 
[[
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
]], {
  }))



cs("OpenCV_epipolar_example", fmt( -- opencv epipolar example
[[
using namespace std;
using namespace cv;

// pose estimation 2d2d epipolar geometry

void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

void pose_estimation_2d2d(
  std::vector<KeyPoint> keypoints_1,
  std::vector<KeyPoint> keypoints_2,
  std::vector<DMatch> matches,
  Mat &R, Mat &t);


// projection pixel 2 cam
Point2d pixel2cam(const Point2d &p, const Mat &K);

int main(int argc, char **argv) {{
  if (argc != 3) {{
    cout << "usage: pose_estimation_2d2d img1 img2" << endl;
    return 1;
  }}
  
  //-- read images
  Mat img_1 = imread(argv[1]);
  Mat img_2 = imread(argv[2]);
  assert(img_1.data && img_2.data && "Can not load images!");

  // compute matches between images
  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "number of matches : " << matches.size() << endl;

  //--estimate pose
  Mat R, t;
  pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

  //-- compute t skew matrix
  Mat t_x =
    (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
      t.at<double>(2, 0), 0, -t.at<double>(0, 0),
      -t.at<double>(1, 0), t.at<double>(0, 0), 0);

  cout << "t^R=" << endl << t_x * R << endl;

  //-- compute epipolar constraints
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  // display epipolar constraint for each matches
  for (DMatch m: matches) {{
    Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
    Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
    Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
    Mat d = y2.t() * t_x * R * y1;
    cout << "epipolar constraint = " << d << endl;
  }}
  return 0;
}}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {{

  //-- define descriptors
  Mat descriptors_1, descriptors_2;
  // used in OpenCV3
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  // extract keypoints
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);
  // extract descriptors
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);
  // extract match
  vector<DMatch> match;
  matcher->match(descriptors_1, descriptors_2, match);
  // threshold to filter good matches
  double min_dist = 10000, max_dist = 0;
  // get min max distances
  for (int i = 0; i < descriptors_1.rows; i++) {{
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }}
  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);
  // filter good matches
  for (int i = 0; i < descriptors_1.rows; i++) {{
    if (match[i].distance <= max(2 * min_dist, 30.0)) {{
      matches.push_back(match[i]);
    }}
  }}
}}

// pixel input normalized coords output
Point2d pixel2cam(const Point2d &p, const Mat &K) {{
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}}

// pose estimation epipolar constraintes
void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,
                          std::vector<KeyPoint> keypoints_2,
                          std::vector<DMatch> matches,
                          Mat &R, Mat &t) {{
  // intrinsics
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

  //-- 2D points matched on both images
  vector<Point2f> points1;
  vector<Point2f> points2;

  // gather data from keypoints and matches
  for (int i = 0; i < (int) matches.size(); i++) {{
    points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }}

  //-- Compute fundamental matrix K-T*t^*R*K-1
  Mat fundamental_matrix;
  // find F with 8 points
  fundamental_matrix = findFundamentalMat(points1, points2, FM_8POINT);

  cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

  //-- intrinsics
  Point2d principal_point(325.1, 249.7); // cx cy
  // fx fy
  double focal_length = 521;
  Mat essential_matrix;

  // compute essential matrix
  essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
  cout << "essential_matrix is " << endl << essential_matrix << endl;

  //--compute homography
  Mat homography_matrix;
  // Compute H using RANSAC random 3 points
  homography_matrix = findHomography(points1, points2, RANSAC, 3);
  cout << "homography_matrix is " << endl << homography_matrix << endl;

  //-- get pose from essential matrix
  recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
  cout << "R is " << endl << R << endl;
  cout << "t is " << endl << t << endl;
}}
]], {
  }))


cs("Opencv_triangulation_example", fmt( -- opencv triangulation example 
[[
// #include "extra.h" // used in opencv2
using namespace std;
using namespace cv;

// find features matches
void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

// pose estimation epipolar constraintes to get R, t
void pose_estimation_2d2d(
  const std::vector<KeyPoint> &keypoints_1,
  const std::vector<KeyPoint> &keypoints_2,
  const std::vector<DMatch> &matches,
  Mat &R, Mat &t);

// compute depth from R, t
void triangulation(
  const vector<KeyPoint> &keypoint_1,
  const vector<KeyPoint> &keypoint_2,
  const std::vector<DMatch> &matches,
  const Mat &R, const Mat &t,
  vector<Point3d> &points
);

/// get color
inline cv::Scalar get_color(float depth) {{
  float up_th = 50, low_th = 10, th_range = up_th - low_th;
  if (depth > up_th) depth = up_th;
  if (depth < low_th) depth = low_th;
  return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}}

// get normalized plane coords from pixels
Point2f pixel2cam(const Point2d &p, const Mat &K);

int main(int argc, char **argv) {{
  if (argc != 3) {{
    cout << "usage: triangulation img1 img2" << endl;
    return 1;
  }}

  //-- load images
  Mat img_1 = imread(argv[1], IMREAD_COLOR);
  Mat img_2 = imread(argv[2], IMREAD_COLOR);

  // compute matches
  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "Number of matches : " << matches.size() << endl;

  //-- Compute Pose
  Mat R, t;
  pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

  //-- Compute Depth
  vector<Point3d> points;
  triangulation(keypoints_1, keypoints_2, matches, R, t, points);

  //-- Display images with matches
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  Mat img1_plot = img_1.clone();
  Mat img2_plot = img_2.clone();
  for (int i = 0; i < matches.size(); i++) {{
    // depth from triangulation
    float depth1 = points[i].z;
    cout << "depth: " << depth1 << endl;
    // get normalized coord 
    Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
    // draw a circle on the image with color corresponding to depth
    cv::circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 2, get_color(depth1), 2);

    // compute transfo of points in image 2
    Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
    float depth2 = pt2_trans.at<double>(2, 0);
    // draw circle on img2 with depth
    cv::circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 2, get_color(depth2), 2);
  }}
  cv::imshow("img 1", img1_plot);
  cv::imshow("img 2", img2_plot);
  cv::waitKey();

  return 0;
}}

// classical function to compute matches
void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {{
  //-- extract features
  Mat descriptors_1, descriptors_2;
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
  // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  vector<DMatch> match;
  matcher->match(descriptors_1, descriptors_2, match);

  double min_dist = 10000, max_dist = 0;

  for (int i = 0; i < descriptors_1.rows; i++) {{
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }}

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  for (int i = 0; i < descriptors_1.rows; i++) {{
    if (match[i].distance <= max(2 * min_dist, 30.0)) {{
      matches.push_back(match[i]);
    }}
  }}
}}

void pose_estimation_2d2d(
  const std::vector<KeyPoint> &keypoints_1,
  const std::vector<KeyPoint> &keypoints_2,
  const std::vector<DMatch> &matches,
  Mat &R, Mat &t) {{
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

  vector<Point2f> points1;
  vector<Point2f> points2;

  for (int i = 0; i < (int) matches.size(); i++) {{
    points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }}

  Point2d principal_point(325.1, 249.7);
  int focal_length = 521;

  Mat essential_matrix;
  essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);

  recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
}}


// triangulation
void triangulation(
  const vector<KeyPoint> &keypoint_1,
  const vector<KeyPoint> &keypoint_2,
  const std::vector<DMatch> &matches,
  const Mat &R, const Mat &t,
  vector<Point3d> &points) {{
    // Pose img1 I
    Mat T1 = (Mat_<float>(3, 4) <<
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0);
    // Pose img2 R, t
    Mat T2 = (Mat_<float>(3, 4) <<
    R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
    R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
    R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
    );
    // camera intrinsics
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    // gather matches
    vector<Point2f> pts_1, pts_2;
    for (DMatch m:matches) {{
        pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
        pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
    }}

    // triangulate points
    Mat pts_4d;
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    // gather results convert to non homogeneous coords
    for (int i = 0; i < pts_4d.cols; i++) {{
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0); // divide by 4th coef to convert to non homogeneous coords
        Point3d p(
                x.at<float>(0, 0),
                x.at<float>(1, 0),
                x.at<float>(2, 0)
    );
    points.push_back(p);
    }}
}}

// project pixel to normalized coord plane
Point2f pixel2cam(const Point2d &p, const Mat &K) {{
  return Point2f
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}}
]], {
  }))


cs("G2O_ba_include_packages", fmt( -- g2o include package
[[
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
]], {
  }))


cs("G2O_pnp_example", fmt( -- g2o ba example
[[
using namespace std;
using namespace cv;

// find feature matches
void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

// projection pixels
Point2d pixel2cam(const Point2d &p, const Mat &K);

// typedef for convenience
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

// BA with G2O
void bundleAdjustmentG2O(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose
);

// BA with GN
void bundleAdjustmentGaussNewton(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose
);

// main app
int main(int argc, char **argv) {{
  if (argc != 5) {{
    cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
    return 1;
  }}
  //-- read images 
  Mat img_1 = imread(argv[1], IMREAD_COLOR);
  Mat img_2 = imread(argv[2], IMREAD_COLOR);
  assert(img_1.data && img_2.data && "Can not load images!");

  // gather matches
  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "number of images : " << matches.size() << endl;

  // gather 3D points from depth 1  and 2D points from img 2
  Mat d1 = imread(argv[3], IMREAD_UNCHANGED);       // read depth depth image 1
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<Point3f> pts_3d;
  vector<Point2f> pts_2d;
  // go through matches
  for (DMatch m:matches) {{
      // get the depth of pixel  x y we start with y row and x col 
      ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
      if (d == 0)   // bad depth
          continue;
      // divide by 5000
      float dd = d / 5000.0;
      // project pixel to normalized coord plane
      Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
      // we gather data
      pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
      pts_2d.push_back(keypoints_2[m.trainIdx].pt);
  }}

  cout << "3d-2d pairs: " << pts_3d.size() << endl;

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  Mat r, t;
  solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // solve pnp with opencv
  Mat R;
  cv::Rodrigues(r, R); // use rodrigues to convert r rotation vector to R rotation matrix
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;

  cout << "R=" << endl << R << endl;
  cout << "t=" << endl << t << endl;

  // convert data to eigen format
  VecVector3d pts_3d_eigen;
  VecVector2d pts_2d_eigen;
  for (size_t i = 0; i < pts_3d.size(); ++i) {{
    pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
    pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
  }}

  cout << "calling bundle adjustment by gauss newton" << endl;
  Sophus::SE3d pose_gn;
  t1 = chrono::steady_clock::now();
  bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp by gauss newton cost time: " << time_used.count() << " seconds." << endl;

  cout << "calling bundle adjustment by g2o" << endl;
  Sophus::SE3d pose_g2o;
  t1 = chrono::steady_clock::now();
  bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp by g2o cost time: " << time_used.count() << " seconds." << endl;
  return 0;
}}


// classic function to compute matches
void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {{
    // compute descriptors
  Mat descriptors_1, descriptors_2;
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  vector<DMatch> match;
  matcher->match(descriptors_1, descriptors_2, match);

  double min_dist = 10000, max_dist = 0;
  for (int i = 0; i < descriptors_1.rows; i++) {{
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }}

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  for (int i = 0; i < descriptors_1.rows; i++) {{
    if (match[i].distance <= max(2 * min_dist, 30.0)) {{
      matches.push_back(match[i]);
    }}
  }}
}}

// project pixel to normalized coord
Point2d pixel2cam(const Point2d &p, const Mat &K) {{
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}}


// BA with GN
void bundleAdjustmentGaussNewton(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose) 
{{
    // typedef for convenience 6d for pose in vector space lie alegbra se3
  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  const int iterations = 10;
  double cost = 0, lastCost = 0;
  // get intrinsics
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);

  // solve GN
  for (int iter = 0; iter < iterations; iter++) {{
      // Hessian matrix
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    // lie algebra
    Vector6d b = Vector6d::Zero();

    cost = 0;
    // compute cost for each point
    for (int i = 0; i < points_3d.size(); i++) {{
      Eigen::Vector3d pc = pose * points_3d[i];
      double inv_z = 1.0 / pc[2];
      double inv_z2 = inv_z * inv_z;
      Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);

      Eigen::Vector2d e = points_2d[i] - proj;

      cost += e.squaredNorm();
      Eigen::Matrix<double, 2, 6> J;
      // compute J for pose
      J << -fx * inv_z,
        0,
        fx * pc[0] * inv_z2,
        fx * pc[0] * pc[1] * inv_z2,
        -fx - fx * pc[0] * pc[0] * inv_z2,
        fx * pc[1] * inv_z,
        0,
        -fy * inv_z,
        fy * pc[1] * inv_z2,
        fy + fy * pc[1] * pc[1] * inv_z2,
        -fy * pc[0] * pc[1] * inv_z2,
        -fy * pc[0] * inv_z;

      H += J.transpose() * J;
      b += -J.transpose() * e;
    }}

    Vector6d dx;
    dx = H.ldlt().solve(b);

    if (isnan(dx[0])) {{
      cout << "result is nan!" << endl;
      break;
    }}

    if (iter > 0 && cost >= lastCost) {{
      // cost increase, update is not good
      cout << "cost: " << cost << ", last cost: " << lastCost << endl;
      break;
    }}

    // update your estimation
    pose = Sophus::SE3d::exp(dx) * pose;
    lastCost = cost;

    cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << endl;
    if (dx.norm() < 1e-6) {{
      // converge
      break;
    }}
  }}

  cout << "pose by g-n: \n" << pose.matrix() << endl;
}}


/// vertex and edges used in g2o ba we optimize the pose only
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  virtual void setToOriginImpl() override {{
    _estimate = Sophus::SE3d();
  }}

  /// left multiplication on SE3
  virtual void oplusImpl(const double *update) override {{
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
  }}

  virtual bool read(istream &in) override {{}}

  virtual bool write(ostream &out) const override {{}}
}};


// edge definition we optimize pose 
class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  // constructor edge
  EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K) {{}}

  // error corresponding to the edge
  virtual void computeError() override {{
    const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
    pos_pixel /= pos_pixel[2]; // pixel we divide by the third coord to normalize
    _error = _measurement - pos_pixel.head<2>(); // only consider u v not the third coord
  }}

  // define jacobian to optimize
  virtual void linearizeOplus() override {{
    const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d pos_cam = T * _pos3d;
    double fx = _K(0, 0);
    double fy = _K(1, 1);
    double cx = _K(0, 2);
    double cy = _K(1, 2);
    double X = pos_cam[0];
    double Y = pos_cam[1];
    double Z = pos_cam[2];
    double Z2 = Z * Z;
    _jacobianOplusXi
      << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
      0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
  }}

  virtual bool read(istream &in) override {{}}

  virtual bool write(ostream &out) const override {{}}

private:
  Eigen::Vector3d _pos3d;
  Eigen::Matrix3d _K;
}};


// BA with G2O define graph etc
void bundleAdjustmentG2O(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose) {{

    // define block solver 
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;  // pose is 6, landmark is 3
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; 

  // Define type of optimization algo
  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
    std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;     // set up optimizer
  optimizer.setAlgorithm(solver);
  optimizer.setVerbose(true); // print infos

  // vertex related to pose a single vertex a single pose to optimize here
  VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
  vertex_pose->setId(0);
  vertex_pose->setEstimate(Sophus::SE3d());
  optimizer.addVertex(vertex_pose);

  // K of intrinsics
  Eigen::Matrix3d K_eigen;
  K_eigen <<
          K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
    K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
    K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

  // edges between pose and each features
  int index = 1;
  for (size_t i = 0; i < points_2d.size(); ++i) {{
    auto p2d = points_2d[i];
    auto p3d = points_3d[i];
    EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
    edge->setId(index);
    edge->setVertex(0, vertex_pose);
    edge->setMeasurement(p2d);
    edge->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(edge);
    index++;
  }}

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.setVerbose(true);
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
  cout << "pose estimated by g2o =\n" << vertex_pose->estimate().matrix() << endl;
  pose = vertex_pose->estimate();
}}
]], {
  }))



cs("G2O_icp_example", fmt( -- g2o icp example 
[[
using namespace std;
using namespace cv;

// function to compute matches
void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

// projection pixel to cam
Point2d pixel2cam(const Point2d &p, const Mat &K);

// ICP pose estimation from 3D 3D
void pose_estimation_3d3d(
  const vector<Point3f> &pts1,
  const vector<Point3f> &pts2,
  Mat &R, Mat &t
);

// BA 
void bundleAdjustment(
  const vector<Point3f> &points_3d,
  const vector<Point3f> &points_2d,
  Mat &R, Mat &t
);


/// vertex and edges used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  virtual void setToOriginImpl() override {{
    _estimate = Sophus::SE3d();
  }}

  /// left multiplication on SE3
  virtual void oplusImpl(const double *update) override {{
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
  }}

  virtual bool read(istream &in) override {{}}

  virtual bool write(ostream &out) const override {{}}
}};


/// g2o edge for to compute error
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose> {{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d &point) : _point(point) {{}}

  // error corresponding to the edge here pt3d(I1) - T*pt3d(I2)
  virtual void computeError() override {{
    const VertexPose *pose = static_cast<const VertexPose *> ( _vertices[0] );
    _error = _measurement - pose->estimate() * _point;
  }}

  // jacobian to optimize
  virtual void linearizeOplus() override {{
    VertexPose *pose = static_cast<VertexPose *>(_vertices[0]);
    Sophus::SE3d T = pose->estimate();
    // pts transfo
    Eigen::Vector3d xyz_trans = T * _point;
    // jacobian
    _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans);
  }}

  bool read(istream &in) {{}}

  bool write(ostream &out) const {{}}

protected:
  Eigen::Vector3d _point;
}};

// main app
int main(int argc, char **argv) {{
  if (argc != 5) {{
    cout << "usage: pose_estimation_3d3d img1 img2 depth1 depth2" << endl;
    return 1;
  }}
  //-- read images
  Mat img_1 = imread(argv[1], IMREAD_COLOR);
  Mat img_2 = imread(argv[2], IMREAD_COLOR);

  // extract features
  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "features matches : " << matches.size() << endl;

  // load depths
  Mat depth1 = imread(argv[3], IMREAD_UNCHANGED);       // depth1
  Mat depth2 = imread(argv[4], IMREAD_UNCHANGED);       // depth2
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<Point3f> pts1, pts2;

  for (DMatch m:matches) {{
    ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
    if (d1 == 0 || d2 == 0)   // bad depth
      continue;
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
    float dd1 = float(d1) / 5000.0;
    float dd2 = float(d2) / 5000.0;
    pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
    pts2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));
  }}

  cout << "3d-3d pairs: " << pts1.size() << endl;
  Mat R, t;
  // get pose estimate
  pose_estimation_3d3d(pts1, pts2, R, t);
  cout << "ICP via SVD results: " << endl;
  cout << "R = " << R << endl;
  cout << "t = " << t << endl;
  cout << "R_inv = " << R.t() << endl;
  cout << "t_inv = " << -R.t() * t << endl;

  cout << "calling bundle adjustment" << endl;

  // BA with g2o
  bundleAdjustment(pts1, pts2, R, t);

  // verify p1 = R * p2 + t
  for (int i = 0; i < 5; i++) {{
    cout << "p1 = " << pts1[i] << endl;
    cout << "p2 = " << pts2[i] << endl;
    cout << "(R*p2+t) = " <<
         R * (Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, pts2[i].z) + t
         << endl;
    cout << endl;
  }}
}}




// find features matches
void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {{
  //-- 初始化
  Mat descriptors_1, descriptors_2;
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  vector<DMatch> match;
  matcher->match(descriptors_1, descriptors_2, match);

  double min_dist = 10000, max_dist = 0;

  for (int i = 0; i < descriptors_1.rows; i++) {{
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }}

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  for (int i = 0; i < descriptors_1.rows; i++) {{
    if (match[i].distance <= max(2 * min_dist, 30.0)) {{
      matches.push_back(match[i]);
    }}
  }}
}}

// pixel 2 cam
Point2d pixel2cam(const Point2d &p, const Mat &K) {{
  return Point2d(
    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
  );
}}

// pose estimation 3d3d using linear algebra SVD
void pose_estimation_3d3d(const vector<Point3f> &pts1,
                          const vector<Point3f> &pts2,
                          Mat &R, Mat &t) {{
  Point3f p1, p2;     // center of mass
  int N = pts1.size();
  for (int i = 0; i < N; i++) {{
    p1 += pts1[i];
    p2 += pts2[i];
  }}
  p1 = Point3f(Vec3f(p1) / N);
  p2 = Point3f(Vec3f(p2) / N);
  vector<Point3f> q1(N), q2(N); // remove the center
  for (int i = 0; i < N; i++) {{
    q1[i] = pts1[i] - p1;
    q2[i] = pts2[i] - p2;
  }}

  // compute q1*q2^T
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
  for (int i = 0; i < N; i++) {{
    W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
  }}
  cout << "W=" << W << endl;

  // SVD on W
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  cout << "U=" << U << endl;
  cout << "V=" << V << endl;

  Eigen::Matrix3d R_ = U * (V.transpose());
  if (R_.determinant() < 0) {{
    R_ = -R_;
  }}
  Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

  // convert to cv::Mat
  R = (Mat_<double>(3, 3) <<
    R_(0, 0), R_(0, 1), R_(0, 2),
    R_(1, 0), R_(1, 1), R_(1, 2),
    R_(2, 0), R_(2, 1), R_(2, 2)
  );
  t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}}



// BA with g2o
void bundleAdjustment(
  const vector<Point3f> &pts1,
  const vector<Point3f> &pts2,
  Mat &R, Mat &t) {{
  // define typdef for convenience
  typedef g2o::BlockSolverX BlockSolverType;
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

  auto solver = new g2o::OptimizationAlgorithmLevenberg(
    std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);
  optimizer.setVerbose(true);

  // vertex
  VertexPose *pose = new VertexPose(); // camera pose
  pose->setId(0);
  pose->setEstimate(Sophus::SE3d());
  optimizer.addVertex(pose);

  // edges for all pt3D from img2
  for (size_t i = 0; i < pts1.size(); i++) {{
    EdgeProjectXYZRGBDPoseOnly *edge = new EdgeProjectXYZRGBDPoseOnly(
      Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
    // optimize pose
    edge->setVertex(0, pose);
    // measurment pts3D from img1
    edge->setMeasurement(Eigen::Vector3d(
      pts1[i].x, pts1[i].y, pts1[i].z));
    edge->setInformation(Eigen::Matrix3d::Identity());
    optimizer.addEdge(edge);
  }}

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "optimization costs time: " << time_used.count() << " seconds." << endl;

  cout << endl << "after optimization:" << endl;
  cout << "T=\n" << pose->estimate().matrix() << endl;

  // convert to cv::Mat
  Eigen::Matrix3d R_ = pose->estimate().rotationMatrix();
  Eigen::Vector3d t_ = pose->estimate().translation();
  R = (Mat_<double>(3, 3) <<
    R_(0, 0), R_(0, 1), R_(0, 2),
    R_(1, 0), R_(1, 1), R_(1, 2),
    R_(2, 0), R_(2, 1), R_(2, 2)
  );
  t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}}
]], {
  }))



cs("OpenCV_optical_flow_example", fmt( -- optical flow example 
[[
using namespace std;
using namespace cv;

string file_1 = "../LK1.png";  // first image
string file_2 = "../LK2.png";  // second image

/// Optical flow tracker and interface
class OpticalFlowTracker {{
    public:
        OpticalFlowTracker(
                const Mat &img1_,
                const Mat &img2_,
                const vector<KeyPoint> &kp1_,
                vector<KeyPoint> &kp2_,
                vector<bool> &success_,
                bool inverse_ = true, bool has_initial_ = false) :
            img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_),
            has_initial(has_initial_) {{}}

        void calculateOpticalFlow(const Range &range);

    private:
        const Mat &img1;
        const Mat &img2;
        const vector<KeyPoint> &kp1;
        vector<KeyPoint> &kp2;
        vector<bool> &success;
        bool inverse = true;
        bool has_initial = false;
}};



/**
 * single level optical flow
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse use inverse formulation?
 */
void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false,
    bool has_initial_guess = false
);



/**
 * multi level optical flow, scale of pyramid is set to 2 by default
 * the image pyramid will be create inside the function
 * @param [in] img1 the first pyramid
 * @param [in] img2 the second pyramid
 * @param [in] kp1 keypoints in img1
 * @param [out] kp2 keypoints in img2
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse set true to enable inverse formulation
 */
void OpticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false
);



/**
 * get a gray scale value from reference image (bi-linear interpolated)
 * @param img
 * @param x
 * @param y
 * @return the interpolated value of this pixel
 */

inline float GetPixelValue(const cv::Mat &img, float x, float y) {{
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;
    
    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);
    
    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)
    + xx * (1 - yy) * img.at<uchar>(y, x_a1)
    + (1 - xx) * yy * img.at<uchar>(y_a1, x)
    + xx * yy * img.at<uchar>(y_a1, x_a1);
}}



// main app
int main(int argc, char **argv) {{

    // images, note they are CV_8UC1, not CV_8UC3
    Mat img1 = imread(file_1, 0);
    Mat img2 = imread(file_2, 0);

    // key points, using GFTT here. GFFT features extractor very fast
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
    detector->detect(img1, kp1);

    // now lets track these key points in the second image
    // first use single level LK in the validation picture
    vector<KeyPoint> kp2_single;
    vector<bool> success_single;
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);

    // then test multi-level LK
    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    // we use inverse method here 
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by gauss-newton: " << time_used.count() << endl;

    // use opencv's flow for validation
    vector<Point2f> pt1, pt2;
    for (auto &kp: kp1) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by opencv: " << time_used.count() << endl;

    // plot the differences of those functions
    Mat img2_single;
    cv::cvtColor(img2, img2_single, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_single.size(); i++) {{
        if (success_single[i]) {{
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
        }}
    }}

    Mat img2_multi;
    cv::cvtColor(img2, img2_multi, COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); i++) {{
        if (success_multi[i]) {{
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }}
    }}

    Mat img2_CV;
    cv::cvtColor(img2, img2_CV, COLOR_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {{
        if (status[i]) {{
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }}
    }}

    cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multi level", img2_multi);
    cv::imshow("tracked by opencv", img2_CV);
    cv::waitKey(0);

    return 0;
}}




// optical flow single level parallel for
void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse, bool has_initial) 
{{
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);
    parallel_for_(Range(0, kp1.size()),
                  std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));
}}


// optical flow calculation
void OpticalFlowTracker::calculateOpticalFlow(const Range &range) 
{{
    // parameters
    int half_patch_size = 4;
    int iterations = 10;
    // go through features for each feature patch of 4
    for (size_t i = range.start; i < range.end; i++) {{
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx,dy need to be estimated
        if (has_initial) {{
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }}

        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded

        // Gauss-Newton iterations
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();    // hessian
        Eigen::Vector2d b = Eigen::Vector2d::Zero();    // bias
        Eigen::Vector2d J;  // jacobian
        for (int iter = 0; iter < iterations; iter++) {{

            if (inverse == false) {{
                H = Eigen::Matrix2d::Zero(); // not inverse H is reset and updated
                b = Eigen::Vector2d::Zero();
            }} else {{
                // only reset b
                b = Eigen::Vector2d::Zero(); // H do not change with inverse method
            }}

            cost = 0;

            // compute cost and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {{
                    // photometric error
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                                   GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);;  // Jacobian
                    if (inverse == false) {{
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                   GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                                   GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))
                        );
                    }} else if (iter == 0) {{
                        // in inverse mode, J keeps same for all iterations
                        // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                   GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                            0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                   GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1))
                        );
                    }}
                    // compute H, b and set cost;
                    b += -error * J;
                    cost += error * error;
                    if (inverse == false || iter == 0) {{
                        // also update H
                        H += J * J.transpose();
                    }}
                }}

            // compute update
            Eigen::Vector2d update = H.ldlt().solve(b);

            if (std::isnan(update[0])) {{
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }}

            // cost increase we stop
            if (iter > 0 && cost > lastCost) {{
                break;
            }}

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;

            if (update.norm() < 1e-2) {{
                // converge
                break;
            }}
        }}

        success[i] = succ;

        // set kp2
        kp2[i].pt = kp.pt + Point2f(dx, dy);
    }}
}}



// optical flow multilevel
void OpticalFlowMultiLevel
(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse) {{

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {{1.0, 0.5, 0.25, 0.125}};

    // create pyramids
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    vector<Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++) {{
        if (i == 0) {{
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        }} else {{
            Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }}
    }}
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "build pyramid time: " << time_used.count() << endl;

    // coarse-to-fine LK tracking in pyramids
    vector<KeyPoint> kp1_pyr, kp2_pyr;
    for (auto &kp:kp1) {{
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids - 1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }}

    for (int level = pyramids - 1; level >= 0; level--) {{
        // from coarse to fine
        success.clear();
        t1 = chrono::steady_clock::now();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);
        t2 = chrono::steady_clock::now();
        auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        cout << "track pyr " << level << " cost time: " << time_used.count() << endl;

        if (level > 0) {{
            for (auto &kp: kp1_pyr)
                kp.pt /= pyramid_scale;
            for (auto &kp: kp2_pyr)
                kp.pt /= pyramid_scale;
        }}
    }}

    for (auto &kp: kp2_pyr)
        kp2.push_back(kp);
}}
]], {
  }))




cs("Opencv_direct_method_example", fmt( -- opencv direct method example 
[[
using namespace std;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// Camera intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline
double baseline = 0.573;
// paths
string left_file = "../left.png";
string disparity_file = "../disparity.png";
boost::format fmt_others("../%06d.png");    // other files


// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;


/// class for accumulator jacobians in parallel
class JacobianAccumulator {{
public:
    JacobianAccumulator(
        const cv::Mat &img1_,
        const cv::Mat &img2_,
        const VecVector2d &px_ref_,
        const vector<double> depth_ref_,
        Sophus::SE3d &T21_) :
        img1(img1_), img2(img2_), px_ref(px_ref_), depth_ref(depth_ref_), T21(T21_) {{
        projection = VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0));
    }}

    /// accumulate jacobians in a range
    void accumulate_jacobian(const cv::Range &range);

    /// get hessian matrix
    Matrix6d hessian() const {{ return H; }}

    /// get bias
    Vector6d bias() const {{ return b; }}

    /// get total cost
    double cost_func() const {{ return cost; }}

    /// get projected points
    VecVector2d projected_points() const {{ return projection; }}

    /// reset h, b, cost to zero
    void reset() {{
        H = Matrix6d::Zero();
        b = Vector6d::Zero();
        cost = 0;
    }}

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const VecVector2d &px_ref;
    const vector<double> depth_ref;
    Sophus::SE3d &T21;
    VecVector2d projection; // projected points

    std::mutex hessian_mutex;
    Matrix6d H = Matrix6d::Zero();
    Vector6d b = Vector6d::Zero();
    double cost = 0;
}};



/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21
);


/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21
);


// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {{
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    // img.step return number of bytes each row occupies
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}}


// main app
int main(int argc, char **argv) {{

    // load img and depth
    cv::Mat left_img = cv::imread(left_file, 0);
    cv::Mat disparity_img = cv::imread(disparity_file, 0);

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 2000;
    int boarder = 20;
    VecVector2d pixels_ref;
    vector<double> depth_ref;

    // generate pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++) {{
        int x = rng.uniform(boarder, left_img.cols - boarder);  // don't pick pixels close to boarder
        int y = rng.uniform(boarder, left_img.rows - boarder);  // don't pick pixels close to boarder
        int disparity = disparity_img.at<uchar>(y, x);
        double depth = fx * baseline / disparity; // you know this is disparity to depth
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }}

    // estimates 01~05.png's pose using this information
    Sophus::SE3d T_cur_ref;

    for (int i = 1; i < 6; i++) {{  // 1~10
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);
        // try single layer by uncomment this line
        DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
        //DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
    }}
    return 0;
}}



// pose estimation single layer
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21) {{

    const int iterations = 10;
    double cost = 0, lastCost = 0;
    auto t1 = chrono::steady_clock::now();
    JacobianAccumulator jaco_accu(img1, img2, px_ref, depth_ref, T21);

    for (int iter = 0; iter < iterations; iter++) {{
        jaco_accu.reset();
        // on compute le probleme H*update=b to compute update
        cv::parallel_for_(cv::Range(0, px_ref.size()),
                          std::bind(&JacobianAccumulator::accumulate_jacobian, &jaco_accu, std::placeholders::_1));
        Matrix6d H = jaco_accu.hessian();
        Vector6d b = jaco_accu.bias();

        // solve update and put it into estimation
        Vector6d update = H.ldlt().solve(b);;
        // update pose
        T21 = Sophus::SE3d::exp(update) * T21;
        cost = jaco_accu.cost_func();

        if (std::isnan(update[0])) {{
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }}
        if (iter > 0 && cost > lastCost) {{
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }}
        if (update.norm() < 1e-3) {{
            // converge
            break;
        }}

        lastCost = cost;
        cout << "iteration: " << iter << ", cost: " << cost << endl;
    }}

    cout << "T21 = \n" << T21.matrix() << endl;
    auto t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "direct method for single layer: " << time_used.count() << endl;

    // plot the projected pixels here
    cv::Mat img2_show;
    cv::cvtColor(img2, img2_show, cv::COLOR_GRAY2BGR);
    VecVector2d projection = jaco_accu.projected_points();
    for (size_t i = 0; i < px_ref.size(); ++i) {{
        auto p_ref = px_ref[i];
        auto p_cur = projection[i];
        if (p_cur[0] > 0 && p_cur[1] > 0) {{
            cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),
                     cv::Scalar(0, 250, 0));
        }}
    }}
    cv::imshow("current", img2_show);
    cv::waitKey();
}}



// accumulate jacobian
void JacobianAccumulator::accumulate_jacobian(const cv::Range &range) {{

    // parameters
    const int half_patch_size = 1;
    int cnt_good = 0;
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;

    // go through data points for which we have pixel and corresponding 3D point
    for (size_t i = range.start; i < range.end; i++) {{

        // compute the projection in the second image
        Eigen::Vector3d point_ref =
            depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - cx) / fx, (px_ref[i][1] - cy) / fy, 1);
        Eigen::Vector3d point_cur = T21 * point_ref;
        if (point_cur[2] < 0)   // depth invalid
            continue;

        float u = fx * point_cur[0] / point_cur[2] + cx, v = fy * point_cur[1] / point_cur[2] + cy;
        if (u < half_patch_size || u > img2.cols - half_patch_size || v < half_patch_size ||
            v > img2.rows - half_patch_size)
            continue;

        projection[i] = Eigen::Vector2d(u, v);
        double X = point_cur[0], Y = point_cur[1], Z = point_cur[2],
            Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;
        cnt_good++;

        // and compute error and jacobian
        for (int x = -half_patch_size; x <= half_patch_size; x++)
            for (int y = -half_patch_size; y <= half_patch_size; y++) {{

                double error = GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) -
                               GetPixelValue(img2, u + x, v + y);
                Matrix26d J_pixel_xi;
                Eigen::Vector2d J_img_pixel;

                J_pixel_xi(0, 0) = fx * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fx * X * Z2_inv;
                J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fx * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fy * Z_inv;
                J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fy * X * Z_inv;

                J_img_pixel = Eigen::Vector2d(
                    0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
                    0.5 * (GetPixelValue(img2, u + x, v + 1 + y) - GetPixelValue(img2, u + x, v - 1 + y))
                );

                // total jacobian
                Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

                hessian += J * J.transpose();
                bias += -error * J;
                cost_tmp += error * error;
            }}
    }}

    if (cnt_good) {{
        // set hessian, bias and cost
        unique_lock<mutex> lck(hessian_mutex);
        H += hessian;
        b += bias;
        cost += cost_tmp / cnt_good;
    }}
}}


// multilayer
void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21) {{

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {{1.0, 0.5, 0.25, 0.125}};

    // create pyramids
    vector<cv::Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++) {{
        if (i == 0) {{
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        }} else {{
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }}
    }}

    double fxG = fx, fyG = fy, cxG = cx, cyG = cy;  // backup the old values
    for (int level = pyramids - 1; level >= 0; level--) {{
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level
        for (auto &px: px_ref) {{
            px_ref_pyr.push_back(scales[level] * px);
        }}

        // scale fx, fy, cx, cy in different pyramid levels
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
    }}

}}
]], {
  }))





cs("G20_pose_graph_cmake", fmt( -- g2O cmake to compile example 
[[
cmake_minimum_required(VERSION 2.8)
project(pose_graph)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

# Eigen
include_directories("/usr/include/eigen3")

option(USE_UBUNTU_20 "Set to ON if you are using Ubuntu 20.04" ON)
if(USE_UBUNTU_20)
    message("You are using Ubuntu 20.04, fmt::fmt will be linked")
    find_package(fmt REQUIRED)
    set(FMT_LIBRARIES fmt::fmt)
endif()

# sophus 
find_package(Sophus REQUIRED)
include_directories(${{Sophus_INCLUDE_DIRS}})

# g2o 
find_package(G2O REQUIRED)
include_directories(${{G2O_INCLUDE_DIRS}})

add_executable(pose_graph_g2o_SE3 pose_graph_g2o_SE3.cpp)
target_link_libraries(pose_graph_g2o_SE3
        g2o_core g2o_stuff g2o_types_slam3d ${{CHOLMOD_LIBRARIES}}
        ${{FMT_LIBRARIES}}
        )
]], {
  }))



cs("G2O_pose_graph_example", fmt( -- g2o pose graph example 
[[
#include <iostream>
#include <fstream>
#include <string>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

using namespace std;

// pose graph g2o SE3

// main app
int main(int argc, char **argv) {{
    if (argc != 2) {{
        cout << "Usage: pose_graph_g2o_SE3 sphere.g2o" << endl;
        return 1;
    }}
    ifstream fin(argv[1]);
    if (!fin) {{
        cout << "file " << argv[1] << " does not exist." << endl;
        return 1;
    }}

    // g2o solver definition
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    // choose LM method
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;     // set uip optimizer
    optimizer.setAlgorithm(solver);   // set up algo
    optimizer.setVerbose(true);       // print infos

    int vertexCnt = 0, edgeCnt = 0; // vertex and edge definition
    while (!fin.eof()) {{
        // lecture du fichier sphere.g2o
        string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT") {{
            // vertex definition built in
            g2o::VertexSE3 *v = new g2o::VertexSE3();
            int index = 0;
            fin >> index;
            v->setId(index);
            v->read(fin);
            optimizer.addVertex(v);
            vertexCnt++;
            if (index == 0)
                v->setFixed(true);
        }} else if (name == "EDGE_SE3:QUAT") {{
            // SE3-SE3 built in edge
            g2o::EdgeSE3 *e = new g2o::EdgeSE3();
            int idx1, idx2;     // get vertices indices
            fin >> idx1 >> idx2;
            e->setId(edgeCnt++);
            e->setVertex(0, optimizer.vertices()[idx1]);
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->read(fin);
            optimizer.addEdge(e);
        }}
        if (!fin.good()) break;
    }}

    cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl;

    cout << "optimizing ..." << endl;
    // start optimization
    optimizer.initializeOptimization();
    // set up number of iterations
    optimizer.optimize(30);

    cout << "saving optimization results ..." << endl;
    optimizer.save("result.g2o");

    return 0;
}}
]], {
  }))




cs("G2O_bundle_adjustment_example", fmt( -- g2o bundle adjustment example
[[
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>
#include <iostream>

// include to formulate the BAL problem
#include "common.h"
#include "sophus/se3.hpp"

using namespace Sophus;
using namespace Eigen;
using namespace std;

// Bundle adjustment with g2o


// define pose and intrinsics
struct PoseAndIntrinsics {{
    PoseAndIntrinsics() {{}}

    /// set from given data address
    explicit PoseAndIntrinsics(double *data_addr) {{
        rotation = SO3d::exp(Vector3d(data_addr[0], data_addr[1], data_addr[2]));
        translation = Vector3d(data_addr[3], data_addr[4], data_addr[5]);
        focal = data_addr[6];
        k1 = data_addr[7];
        k2 = data_addr[8];
    }}

    // set function
    void set_to(double *data_addr) {{
        auto r = rotation.log();
        for (int i = 0; i < 3; ++i) data_addr[i] = r[i];
        for (int i = 0; i < 3; ++i) data_addr[i + 3] = translation[i];
        data_addr[6] = focal;
        data_addr[7] = k1;
        data_addr[8] = k2;
    }}

    // rotation
    SO3d rotation;
    // translation
    Vector3d translation = Vector3d::Zero();
    // intrinsics
    double focal = 0;
    double k1 = 0, k2 = 0;
}};

// vertex corresponding to cameras Vertex defintions 9 pose + intrinsics opimizable variables
class VertexPoseAndIntrinsics : public g2o::BaseVertex<9, PoseAndIntrinsics> {{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoseAndIntrinsics() {{}}

    // get initial values for optimizable variables
    virtual void setToOriginImpl() override {{
        _estimate = PoseAndIntrinsics();
    }}

    // updater 
    virtual void oplusImpl(const double *update) override {{
        _estimate.rotation = SO3d::exp(Vector3d(update[0], update[1], update[2])) * _estimate.rotation;
        _estimate.translation += Vector3d(update[3], update[4], update[5]);
        _estimate.focal += update[6];
        _estimate.k1 += update[7];
        _estimate.k2 += update[8];
    }}

    // projection point to camera
    Vector2d project(const Vector3d &point) {{
        Vector3d pc = _estimate.rotation * point + _estimate.translation;
        pc = -pc / pc[2];
        double r2 = pc.squaredNorm();
        double distortion = 1.0 + r2 * (_estimate.k1 + _estimate.k2 * r2);
        return Vector2d(_estimate.focal * distortion * pc[0],
                        _estimate.focal * distortion * pc[1]);
    }}

    // read and write functions
    virtual bool read(istream &in) {{}}
    virtual bool write(ostream &out) const {{}}
}};

// vertex corresponding to landmarks
class VertexPoint : public g2o::BaseVertex<3, Vector3d> {{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoint() {{}}

    // set estimate to zero
    virtual void setToOriginImpl() override {{
        _estimate = Vector3d(0, 0, 0);
    }}
    // updater
    virtual void oplusImpl(const double *update) override {{
        _estimate += Vector3d(update[0], update[1], update[2]);
    }}
    // read and write functions
    virtual bool read(istream &in) {{}}
    virtual bool write(ostream &out) const {{}}
}};


// edge between cameras and landmarks 2 because the error is on the pixel reprojection error 2d
class EdgeProjection :
    public g2o::BaseBinaryEdge<2, Vector2d, VertexPoseAndIntrinsics, VertexPoint> {{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // compute error pixel reprojection
    virtual void computeError() override {{
        auto v0 = (VertexPoseAndIntrinsics *) _vertices[0];
        auto v1 = (VertexPoint *) _vertices[1];
        auto proj = v0->project(v1->estimate());
        _error = proj - _measurement;
    }}

    // use numeric derivatives
    // we do not define the jacobian here
    
    // read and write functions
    virtual bool read(istream &in) {{}}
    virtual bool write(ostream &out) const {{}}

}};

// solve BA function
void SolveBA(BALProblem &bal_problem);

// main application
int main(int argc, char **argv) {{

    if (argc != 2) {{
        cout << "usage: bundle_adjustment_g2o bal_data.txt" << endl;
        return 1;
    }}

    BALProblem bal_problem(argv[1]);
    // normalize data
    bal_problem.Normalize();
    // perturb data
    bal_problem.Perturb(0.1, 0.5, 0.5);
    // write initial noisy ply
    bal_problem.WriteToPLYFile("initial.ply");
    // solve BA problem
    SolveBA(bal_problem);
    // write final ply
    bal_problem.WriteToPLYFile("final.ply");

    return 0;
}}

// BAL problem
void SolveBA(BALProblem &bal_problem) {{
    // landmarks block
    const int point_block_size = bal_problem.point_block_size();
    // camera block
    const int camera_block_size = bal_problem.camera_block_size();
    double *points = bal_problem.mutable_points();
    double *cameras = bal_problem.mutable_cameras();

    // pose dimension 9, landmark is 3 each edge connect pose to landmark
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    // use LM
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    /// build g2o problem
    const double *observations = bal_problem.observations();
    // vertex
    vector<VertexPoseAndIntrinsics *> vertex_pose_intrinsics;
    vector<VertexPoint *> vertex_points;
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {{
        VertexPoseAndIntrinsics *v = new VertexPoseAndIntrinsics();
        double *camera = cameras + camera_block_size * i;
        v->setId(i);
        v->setEstimate(PoseAndIntrinsics(camera));
        optimizer.addVertex(v);
        vertex_pose_intrinsics.push_back(v);
    }}
    for (int i = 0; i < bal_problem.num_points(); ++i) {{
        VertexPoint *v = new VertexPoint();
        double *point = points + point_block_size * i;
        v->setId(i + bal_problem.num_cameras());
        v->setEstimate(Vector3d(point[0], point[1], point[2]));
        v->setMarginalized(true);
        optimizer.addVertex(v);
        vertex_points.push_back(v);
    }}

    // edge
    for (int i = 0; i < bal_problem.num_observations(); ++i) {{
        EdgeProjection *edge = new EdgeProjection;
        edge->setVertex(0, vertex_pose_intrinsics[bal_problem.camera_index()[i] ]);
        edge->setVertex(1, vertex_points[bal_problem.point_index()[i] ]);
        edge->setMeasurement(Vector2d(observations[2 * i + 0], observations[2 * i + 1]));
        edge->setInformation(Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
    }}

    optimizer.initializeOptimization();
    optimizer.optimize(40);

    // set to bal problem
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {{
        double *camera = cameras + camera_block_size * i;
        auto vertex = vertex_pose_intrinsics[i];
        auto estimate = vertex->estimate();
        estimate.set_to(camera);
    }}
    for (int i = 0; i < bal_problem.num_points(); ++i) {{
        double *point = points + point_block_size * i;
        auto vertex = vertex_points[i];
        for (int k = 0; k < 3; ++k) point[k] = vertex->estimate()[k];
    }}
}}
]], {
  }))



cs("G2O_bundle_adjustment_cmake", fmt( -- g2O bundle adjustment cmake 
[[
cmake_minimum_required(VERSION 2.8)

project(bundle_adjustment)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -std=c++20")

LIST(APPEND CMAKE_MODULE_PATH ${{PROJECT_SOURCE_DIR}}/cmake)

Find_Package(G2O REQUIRED)
Find_Package(Eigen3 REQUIRED)
Find_Package(Sophus REQUIRED)
Find_Package(CSparse REQUIRED)

option(USE_UBUNTU_20 "Set to ON if you are using Ubuntu 20.04" ON)
if(USE_UBUNTU_20)
    message("You are using Ubuntu 20.04, fmt::fmt will be linked")
    find_package(fmt REQUIRED)
    set(FMT_LIBRARIES fmt::fmt)
endif()

SET(G2O_LIBS g2o_csparse_extension g2o_stuff g2o_core cxsparse)

include_directories(${{G2O_INCLUDE_DIRS}})

include_directories(${{PROJECT_SOURCE_DIR}} ${{EIGEN3_INCLUDE_DIR}} ${{CSPARSE_INCLUDE_DIR}})

add_library(bal_common common.cpp)
add_executable(bundle_adjustment_g2o bundle_adjustment_g2o.cpp)
target_link_libraries(bundle_adjustment_g2o ${{G2O_LIBS}} bal_common ${{FMT_LIBRARIES}})
]], {
  }))



cs("DBOW3_feature_training", fmt( -- DBOW3 feature training to train a dictionary for loop closure 
[[
#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

// feature training

int main( int argc, char** argv ) {{
    // read the image 
    cout<<"reading images... "<<endl;
    vector<Mat> images; 
    for ( int i=0; i<10; i++ )
    {{
        string path = "../data/"+to_string(i+1)+".png";
        images.push_back( imread(path) );
    }}
    // detect ORB features
    cout<<"detecting ORB features ... "<<endl;
    Ptr< Feature2D > detector = ORB::create();
    vector<Mat> descriptors;
    for ( Mat& image:images )
    {{
        vector<KeyPoint> keypoints; 
        Mat descriptor;
        detector->detectAndCompute( image, Mat(), keypoints, descriptor );
        descriptors.push_back( descriptor );
    }}
    
    // create vocabulary 
    cout<<"creating vocabulary ... "<<endl;
    DBoW3::Vocabulary vocab;
    vocab.create( descriptors );
    cout<<"vocabulary info: "<<vocab<<endl;
    vocab.save( "vocabulary.yml.gz" );
    cout<<"done"<<endl;
    
    return 0;
}}
]], {
  }))



cs("DBOW3_cmake", fmt( -- DBOW3 cmake to compile code 
[[
cmake_minimum_required( VERSION 2.8 )
project( loop_closure )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# opencv 
find_package( OpenCV REQUIRED )
include_directories( ${{OpenCV_INCLUDE_DIRS}} )

# dbow3 
# dbow3 is a simple lib so I assume you installed it in default directory 
set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
set( DBoW3_LIBS "/usr/local/lib/libDBoW3.so" )

add_executable( feature_training feature_training.cpp )
target_link_libraries( feature_training ${{OpenCV_LIBS}} ${{DBoW3_LIBS}} )
]], {
  }))


-- Tutorial Snippets go here --

-- End Refactoring --

return snippets, autosnippets

