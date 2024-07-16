struct WindowRes {
    uint width;
    uint height;
};

struct Matrices {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 inverse_vp;
    mat4 static_view;
    mat4 cubemap_capture_views[6];
    mat4 cubemap_capture_proj;
};

struct MiscData {
    float debug_number;
    float z_near;
    float z_far;
    uint use_ssao;
    uint use_ibl;
    float light_intensity;
    vec3 light_direction;
    vec3 light_color;
    vec3 camera_pos;
};
