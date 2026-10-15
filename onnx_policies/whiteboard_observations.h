// Whiteboard-drawing observations for unitree_rl_lab (G1 29dof).
//
// Copy to  deploy/robots/g1_29dof/include/whiteboard_observations.h  and add
//   #include "whiteboard_observations.h"
// to  deploy/robots/g1_29dof/src/State_RLBase.cpp  (after the observations.h include).
//
// Terms registered here (names match deploy.yaml):
//   base_lin_vel, pen_pos, pen_vel_est, pen_pos_std, target_pos, foot_contact,
//   pen_board_force.
//
// The policy was trained in a world frame whose origin is the robot's start
// pose and whose +X axis points at the board. Everything below reconstructs
// the same quantities from the G1's IMU and joint encoders only.

#pragma once

#include <algorithm>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace whiteboard
{

using Mat4 = Eigen::Matrix4f;
using Vec3 = Eigen::Vector3f;

inline Mat4 trans(float x, float y, float z)
{
    Mat4 m = Mat4::Identity();
    m(0, 3) = x;
    m(1, 3) = y;
    m(2, 3) = z;
    return m;
}

inline Mat4 rotq(float w, float x, float y, float z)
{
    Mat4 m = Mat4::Identity();
    m.block<3, 3>(0, 0) = Eigen::Quaternionf(w, x, y, z).normalized().toRotationMatrix();
    return m;
}

inline Mat4 rotax(const Vec3 & axis, float angle)
{
    Mat4 m = Mat4::Identity();
    m.block<3, 3>(0, 0) = Eigen::AngleAxisf(angle, axis).toRotationMatrix();
    return m;
}

// Pen tip in the pelvis frame, from the 29 joint angles (SDK/mjlab order).
// Matches g1.xml exactly (verified against MuJoCo to < 1e-6 m).
inline Vec3 pen_in_pelvis(const Eigen::VectorXf & q)
{
    const Vec3 X(1, 0, 0), Y(0, 1, 0), Z(0, 0, 1);
    Mat4 M = rotax(Z, q[12]);                                        // waist_yaw
    M = M * trans(-0.0039635f, 0.f, 0.044f) * rotax(X, q[13]);       // waist_roll
    M = M * rotax(Y, q[14]);                                         // waist_pitch
    M = M * trans(0.0039563f, -0.10021f, 0.24778f)
          * rotq(0.990264f, -0.139201f, 1.38722e-05f, 9.86868e-05f) * rotax(Y, q[22]);  // shoulder_pitch
    M = M * trans(0.f, -0.038f, -0.013831f) * rotq(0.990268f, 0.139172f, 0.f, 0.f) * rotax(X, q[23]);  // shoulder_roll
    M = M * trans(0.f, -0.00624f, -0.1032f) * rotax(Z, q[24]);       // shoulder_yaw
    M = M * trans(0.015783f, 0.f, -0.080518f) * rotax(Y, q[25]);     // elbow
    M = M * trans(0.1f, -0.00188791f, -0.01f) * rotax(X, q[26]);     // wrist_roll
    M = M * trans(0.038f, 0.f, 0.f) * rotax(Y, q[27]);               // wrist_pitch
    M = M * trans(0.046f, 0.f, 0.f) * rotax(Z, q[28]);               // wrist_yaw
    Eigen::Vector4f tip = M * Eigen::Vector4f(0.22f, 0.f, 0.f, 1.f);  // pen tip in wrist_yaw frame
    return tip.head<3>();
}

struct Cfg
{
    bool loaded = false;
    std::string mode = "circle";  // "circle" or "figure8"
    Vec3 pelvis_pos = Vec3(0.083f, 0.039f, 0.775f);  // mean pelvis pose seen in training
    float writing_x = 0.322f;
    float center_y = 0.0f;
    float center_z = 0.80f;
    float radius = 0.12f;
    float spacing = 0.005f;       // circle path point spacing (m)
    float n_points = 152.f;
    float angular_speed = 0.35f;  // figure-8 rad/s
    float approach_s = 3.0f;
    float path_speed = 0.05f;
    float advance_tol = 0.04f;
    float max_tilt_sq = 0.1f;
    float touch_tol_x = 0.010f;   // pen counts as touching the board when x > writing_x - this
    float vel_alpha = 0.3f;
    float pos_std = 0.0066f;
    float board_force = 0.0f;     // value fed for pen_board_force (no sensor on the pen)
};

struct State
{
    long last_k = -1;
    float yaw0 = 0.f;
    float path_s = 0.f;
    bool have_prev = false;
    Vec3 pen = Vec3::Zero();
    Vec3 prev_pen = Vec3::Zero();
    Vec3 vel = Vec3::Zero();
    Vec3 target = Vec3::Zero();
};

inline Cfg & cfg()
{
    static Cfg c;
    return c;
}

inline State & state()
{
    static State s;
    return s;
}

inline void load_cfg(ManagerBasedRLEnv * env)
{
    auto & c = cfg();
    if (c.loaded) return;
    c.loaded = true;
    const YAML::Node n = env->cfg["whiteboard"];
    if (!n.IsDefined() || n.IsNull()) return;
    auto f = [&](const char * k, float & dst) {
        if (n[k].IsDefined()) dst = n[k].as<float>();
    };
    if (n["mode"].IsDefined()) c.mode = n["mode"].as<std::string>();
    if (n["pelvis_pos"].IsDefined()) {
        auto v = n["pelvis_pos"].as<std::vector<float>>();
        if (v.size() == 3) c.pelvis_pos = Vec3(v[0], v[1], v[2]);
    }
    f("writing_x", c.writing_x);
    f("center_y", c.center_y);
    f("center_z", c.center_z);
    f("radius", c.radius);
    f("spacing", c.spacing);
    f("n_points", c.n_points);
    f("angular_speed", c.angular_speed);
    f("approach_s", c.approach_s);
    f("path_speed", c.path_speed);
    f("advance_tol", c.advance_tol);
    f("max_tilt_sq", c.max_tilt_sq);
    f("touch_tol_x", c.touch_tol_x);
    f("vel_alpha", c.vel_alpha);
    f("pos_std", c.pos_std);
    f("board_force", c.board_force);
}

inline float yaw_of(const Eigen::Quaternionf & q)
{
    return std::atan2(2.f * (q.w() * q.z() + q.x() * q.y()), 1.f - 2.f * (q.y() * q.y() + q.z() * q.z()));
}

inline Eigen::Quaternionf safe_quat(const Eigen::Quaternionf & q)
{
    if (q.norm() < 1e-6f) return Eigen::Quaternionf::Identity();
    return q.normalized();
}

// Pen tip in the policy world frame (origin = start pose, +X toward the board).
inline Vec3 compute_pen(ManagerBasedRLEnv * env)
{
    auto & c = cfg();
    auto & s = state();
    const auto q = safe_quat(env->robot->data.root_quat_w);
    const Eigen::Matrix3f R_yaw = Eigen::AngleAxisf(-s.yaw0, Vec3::UnitZ()).toRotationMatrix();
    const Eigen::Matrix3f R = R_yaw * q.toRotationMatrix();
    return c.pelvis_pos + R * pen_in_pelvis(env->robot->data.joint_pos);
}

inline Vec3 target_at(const State & s, float elapsed)
{
    const auto & c = cfg();
    if (c.mode == "figure8") {
        const float phase = std::max(0.f, elapsed - c.approach_s) * c.angular_speed;
        return Vec3(c.writing_x, c.center_y + c.radius * std::sin(phase),
                    c.center_z + 0.5f * c.radius * std::sin(2.f * phase));
    }
    // circle: same path the G-code loader produced (clockwise from the left edge)
    const float phi = s.path_s * c.spacing / c.radius;
    return Vec3(c.writing_x, c.center_y - c.radius * std::cos(phi), c.center_z + c.radius * std::sin(phi));
}

// Advance the hand-built "task state" once per control step.
inline void update(ManagerBasedRLEnv * env)
{
    load_cfg(env);
    auto & c = cfg();
    auto & s = state();
    const long k = env->episode_length;
    if (k != 0 && s.last_k == k) return;
    s.last_k = k;
    const float dt = env->step_dt;
    const float elapsed = k * dt;

    if (k == 0) {
        s.yaw0 = yaw_of(safe_quat(env->robot->data.root_quat_w));
        s.path_s = 0.f;
        s.have_prev = false;
        s.vel.setZero();
    }
    s.pen = compute_pen(env);
    if (s.have_prev) {
        const Vec3 raw = (s.pen - s.prev_pen) / dt;
        s.vel = c.vel_alpha * raw + (1.f - c.vel_alpha) * s.vel;
    }

    if (k > 0 && c.mode == "circle") {
        const auto & g = env->robot->data.projected_gravity_b;
        const bool upright = (g.x() * g.x() + g.y() * g.y()) < c.max_tilt_sq;
        const bool on_target = (s.pen - s.target).norm() < c.advance_tol;
        const bool touching = s.pen.x() > c.writing_x - c.touch_tol_x;
        if (elapsed >= c.approach_s && upright && on_target && touching) {
            const float step = c.path_speed * dt / c.spacing;
            s.path_s = std::min(s.path_s + step, c.n_points - 1.f);
        }
    }
    s.target = target_at(s, elapsed);
    s.prev_pen = s.pen;
    s.have_prev = true;
}

}  // namespace whiteboard

namespace mdp
{

// Velocimeter at the pelvis IMU site. The robot stands still while drawing,
// so the only motion it sees is the IMU lever arm rotating with the pelvis.
REGISTER_OBSERVATION(base_lin_vel)
{
    const Eigen::Vector3f w = env->robot->data.root_ang_vel_b;
    const Eigen::Vector3f r(0.04525f, 0.f, -0.08339f);
    const Eigen::Vector3f v = w.cross(r);
    return std::vector<float>{v.x(), v.y(), v.z()};
}

REGISTER_OBSERVATION(pen_pos)
{
    whiteboard::update(env);
    const auto & p = whiteboard::state().pen;
    return std::vector<float>{p.x(), p.y(), p.z()};
}

REGISTER_OBSERVATION(pen_vel_est)
{
    whiteboard::update(env);
    const auto & v = whiteboard::state().vel;
    return std::vector<float>{v.x(), v.y(), v.z()};
}

REGISTER_OBSERVATION(pen_pos_std)
{
    whiteboard::update(env);
    const float s = whiteboard::cfg().pos_std;
    return std::vector<float>{s, s, s};
}

REGISTER_OBSERVATION(target_pos)
{
    whiteboard::update(env);
    const auto & t = whiteboard::state().target;
    return std::vector<float>{t.x(), t.y(), t.z()};
}

// The G1 has no foot force sensors; it stands on both feet while drawing.
REGISTER_OBSERVATION(foot_contact)
{
    return std::vector<float>{1.0f, 1.0f};
}

REGISTER_OBSERVATION(pen_board_force)
{
    whiteboard::update(env);
    return std::vector<float>{whiteboard::cfg().board_force};
}

}  // namespace mdp
}  // namespace isaaclab
