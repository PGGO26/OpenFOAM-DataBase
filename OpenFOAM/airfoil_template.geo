POINTS

Spline(2000) = {1000:LAST_POINT_INDEX};
Line(2001) = {LAST_POINT_INDEX, 1000};

edge_lc = 0.15;
Point(1900) = { 5, 5, 0, edge_lc};
Point(1901) = { 5, -5, 0, edge_lc};
Point(1902) = { -5, -5, 0, edge_lc};
Point(1903) = { -5, 5, 0, edge_lc};

Line(1) = {1900,1901};
Line(2) = {1901,1902};
Line(3) = {1902,1903};
Line(4) = {1903,1900};

Line Loop (1) = {1,2,3,4};
Line Loop (2) = {2000, 2001};
Plane Surface(1) = {1,2};

Field[1] = BoundaryLayer;
Field[1].EdgesList={2000,2001};
Field[1].FanNodesList = {1000,LAST_POINT_INDEX};
//Field[1].NodesList = {1001,1129};
Field[1].hfar =0.00132;
Field[1].hwall_n = 2e-5; //cell size 1/10 of boundary layer below
Field[1].thickness = 0.00782;
Field[1].ratio = 1.2;
Field[1].Quads =1;
Field[1].AnisoMax= 0.1;
BoundaryLayer Field = 1;



out[] = Extrude {0, 0, 0.5}{
  Surface{1};
  Layers{1};
  Recombine;
};


Printf("%g", #out[]);
Printf("%g", out[0]);
Printf("%g", out[1]);
Printf("%g", out[2]);
Printf("%g", out[3]);
Printf("%g", out[4]);
Printf("%g", out[5]);
Printf("%g", out[6]);
Printf("%g", out[7]);

Physical Surface("back") = {1};
Physical Surface("front") = {out[0]};
Physical Surface("top") = {out[5]};
Physical Surface("exit") = {out[2]};
Physical Surface("bottom") = {out[3]};
Physical Surface("inlet") = {out[4]};
Physical Surface("aerofoil") = {out[6], out[7]};
Physical Volume("internal") = {out[1]};