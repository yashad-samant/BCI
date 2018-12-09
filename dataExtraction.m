
% % LOAD .gdf file in MATLAB
[s,h] = sload('BCICIV_2a_gdf/A01E.gdf');
pos = h.EVENT.POS;
typ = h.EVENT.TYP;
% 
% % data generation
% %final_matrix = [];
% for i = 1:length(pos)
%     if typ(i) == 769 || typ(i) == 770 || typ(i) == 771 || typ(i) == 772
%         final_matrix = vertcat(final_matrix,[s(pos(i), :), typ(i)]);
%     end
%     
% end

%csvwrite('data.csv', final_matrix);

