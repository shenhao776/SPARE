#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pclomp/ndt_omp.h>
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>
using Cloud = pcl::PointCloud<pcl::PointXYZI>;
using NDT = pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>;

struct Job {
  int trial; std::string arm, pcd;
  double sx,sy,sz,sqx,sqy,sqz,sqw;
  double gx,gy,gz,gqx,gqy,gqz,gqw;
  double yawhalf;
};

static double yaw_of(const Eigen::Matrix3d& R) { return std::atan2(R(1,0), R(0,0)); }

static int g_trial = -1;
static std::string g_arm = "?";
static void iter_mark(const char* stage, int idx) {
  static const char* path = std::getenv("NDT_ITER_LOG");
  if (!path) return;
  std::ofstream f(path, std::ios::app);
  f << "#," << stage << "," << g_trial << "," << g_arm << "," << idx << "\n";
}

int main(int argc, char** argv) {
  if (argc < 5) { printf("用法: reloc_seed <map.pcd> <frames_dir> <jobs.txt> <out.csv>\n"); return 2; }
  const std::string map_path = argv[1], dir = argv[2], jobs_path = argv[3], out_path = argv[4];

  pcl::PointCloud<pcl::PointXYZ>::Ptr raw(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile(map_path, *raw) < 0) { printf("map load fail\n"); return 1; }
  Cloud::Ptr map(new Cloud); map->reserve(raw->size());
  for (const auto& p : raw->points) {
    if (!std::isfinite(p.x)) continue;
    pcl::PointXYZI q; q.x=p.x; q.y=p.y; q.z=p.z; q.intensity=0.f; map->push_back(q);
  }
  printf("map: %zu pts\n", map->size()); fflush(stdout);

  NDT::Ptr ndt(new NDT);
  ndt->setResolution(1.0); ndt->setStepSize(0.1);
  ndt->setTransformationEpsilon(0.01); ndt->setNumThreads(4);
  ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
  ndt->setInputTarget(map);
  const int kFullIter = 35;

  std::vector<Job> jobs;
  { std::ifstream f(jobs_path); std::string line;
    while (std::getline(f, line)) { std::istringstream is(line); Job j;
      if (is >> j.trial >> j.arm >> j.pcd >> j.sx >> j.sy >> j.sz
             >> j.sqx >> j.sqy >> j.sqz >> j.sqw
             >> j.gx >> j.gy >> j.gz >> j.gqx >> j.gqy >> j.gqz >> j.gqw
             >> j.yawhalf) jobs.push_back(j); } }
  printf("jobs: %zu\n", jobs.size()); fflush(stdout);
  if (jobs.empty()) return 1;

  std::ofstream out(out_path);
  out << "trial,arm,yawhalf,seed_err_m,converged,fitness,err_m,err_yaw_deg,sec,"
         "fx,fy,fz,fqx,fqy,fqz,fqw\n";
  pcl::VoxelGrid<pcl::PointXYZI> vg; vg.setLeafSize(0.2f,0.2f,0.2f);

  int ok=0, done=0; double tot=0;
  printf("%-5s %-5s %-7s %-9s %-9s %-9s %-8s %s\n",
         "#","arm","yawhalf","seed(m)","err(m)","fitness","dyaw","s");
  for (const Job& j : jobs) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr sr(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile(dir + "/" + j.pcd, *sr) < 0) { printf("skip %s\n", j.pcd.c_str()); continue; }
    Cloud::Ptr s0(new Cloud);
    for (const auto& p : sr->points) {
      double r = std::sqrt(p.x*p.x + p.y*p.y);
      if (!std::isfinite(p.x) || r <= 1.0 || r >= 100.0) continue;
      pcl::PointXYZI q; q.x=p.x; q.y=p.y; q.z=p.z; q.intensity=0.f; s0->push_back(q);
    }
    Cloud::Ptr scan(new Cloud); vg.setInputCloud(s0); vg.filter(*scan);
    if (scan->size() < 3000) { printf("skip %s (稀疏 %zu)\n", j.pcd.c_str(), scan->size()); continue; }

    g_trial = j.trial; g_arm = j.arm;
    Eigen::Quaterniond qs(j.sqw, j.sqx, j.sqy, j.sqz); qs.normalize();
    Eigen::Quaterniond qg(j.gqw, j.gqx, j.gqy, j.gqz); qg.normalize();
    Eigen::Vector3f gt((float)j.gx,(float)j.gy,(float)j.gz);
    Eigen::Matrix4f init = Eigen::Matrix4f::Identity();
    init.block<3,3>(0,0) = qs.toRotationMatrix().cast<float>();
    init(0,3)=(float)j.sx; init(1,3)=(float)j.sy; init(2,3)=(float)j.sz;
    const double seed_err = std::hypot(j.sx-j.gx, j.sy-j.gy);

    auto t0=std::chrono::steady_clock::now();
    constexpr size_t kCoarsePoints=1000;
    constexpr double kYawStepDeg=12.0, kCoarseStep=5.0, kFineStep=2.0, kMaxR=10.0;
    constexpr size_t kSeeds=6, kRefine=16;
    const double half = std::max(6.0, std::min(180.0, j.yawhalf));
    const int kYawBins = (int)std::floor(2.0*half/kYawStepDeg) + 1;
    Cloud::Ptr coarse(new Cloud(*scan));
    if (coarse->size()>kCoarsePoints){ std::mt19937 tr(2026);
      std::shuffle(coarse->points.begin(),coarse->points.end(),tr);
      coarse->points.resize(kCoarsePoints); coarse->width=kCoarsePoints; coarse->height=1; }
    ndt->setInputSource(coarse); ndt->setMaximumIterations(4);
    auto with=[&](double yaw,double dx,double dy){ Eigen::Matrix4f g=init;
      g.block<3,3>(0,0)=Eigen::AngleAxisf((float)yaw,Eigen::Vector3f::UnitZ()).toRotationMatrix()
                        *init.block<3,3>(0,0);
      g(0,3)+=(float)dx; g(1,3)+=(float)dy; return g; };
    auto screen=[&](const Eigen::Matrix4f& g){ iter_mark("coarse", -1); Cloud o; ndt->align(o,g);
      return ndt->hasConverged()?ndt->getFitnessScore():std::numeric_limits<double>::max(); };
    std::vector<std::pair<double,double>> ys;
    for(int i=0;i<kYawBins;++i){ const double deg=-half+i*kYawStepDeg;
      const double y=deg*M_PI/180.0; ys.push_back({screen(with(y,0,0)),y}); }
    std::sort(ys.begin(),ys.end());
    std::vector<std::pair<double,Eigen::Matrix4f>> cand;
    for(const auto& y:ys) for(int ix=-2;ix<=2;++ix) for(int iy=-2;iy<=2;++iy){
      double dx=ix*kCoarseStep,dy=iy*kCoarseStep; if(std::hypot(dx,dy)>kMaxR+1e-6) continue;
      auto g=with(y.second,dx,dy); cand.push_back({screen(g),g}); }
    std::sort(cand.begin(),cand.end(),[](const auto&a,const auto&b){return a.first<b.first;});
    std::vector<std::pair<double,Eigen::Matrix4f>> fine;
    for(size_t i2=0;i2<std::min(kSeeds,cand.size());++i2)
      for(int ix=-1;ix<=1;++ix) for(int iy=-1;iy<=1;++iy){
        Eigen::Matrix4f g=cand[i2].second;
        g(0,3)+=(float)(ix*kFineStep); g(1,3)+=(float)(iy*kFineStep);
        fine.push_back({screen(g),g}); }
    cand.insert(cand.end(),fine.begin(),fine.end());
    std::sort(cand.begin(),cand.end(),[](const auto&a,const auto&b){return a.first<b.first;});
    ndt->setInputSource(scan); ndt->setMaximumIterations(kFullIter);
    double best=std::numeric_limits<double>::max(); Eigen::Matrix4f bg=init;
    for(size_t i=0;i<std::min(kRefine,cand.size());++i){ Cloud o; iter_mark("fine", (int)i); ndt->align(o,cand[i].second);
      if(!ndt->hasConverged()) continue; double sc=ndt->getFitnessScore();
      if(sc<best){ best=sc; bg=ndt->getFinalTransformation(); } }
    Cloud o; iter_mark("final", -1); ndt->align(o,bg);
    const bool conv = ndt->hasConverged();
    Eigen::Matrix4f T=ndt->getFinalTransformation();
    const double fitness = ndt->getFitnessScore();
    double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();

    const double err=(T.block<3,1>(0,3)-gt).norm();
    double dyaw = yaw_of(T.block<3,3>(0,0).cast<double>()) - yaw_of(qg.toRotationMatrix());
    while (dyaw >  M_PI) dyaw -= 2*M_PI;
    while (dyaw < -M_PI) dyaw += 2*M_PI;
    const bool good = conv && err < 1.0;
    ok += good; ++done; tot += sec;
    out << j.trial << "," << j.arm << "," << j.yawhalf << "," << seed_err << ","
        << (conv?1:0) << "," << fitness << "," << err << ","
        << dyaw*180.0/M_PI << "," << sec;
    { Eigen::Quaterniond qf(T.block<3,3>(0,0).cast<double>()); qf.normalize();
      out << "," << T(0,3) << "," << T(1,3) << "," << T(2,3)
          << "," << qf.x() << "," << qf.y() << "," << qf.z() << "," << qf.w(); }
    out << "\n";
    out.flush();
    printf("%-5d %-5s %-7.0f %-9.2f %-9.2f %-9.3f %-8.1f %.1f%s\n", j.trial, j.arm.c_str(),
           j.yawhalf, seed_err, err, fitness, dyaw*180.0/M_PI, sec, good?"  OK":"  FAIL");
    fflush(stdout);
  }
  printf("\n收敛到 1 m 内: %d/%d = %.0f%%   平均 %.1f s/次\n", ok, done, 100.0*ok/done, tot/done);
  return 0;
}
