# Pre-trained models
Released pre-trained models for all main paper experiments.
## Pascal3D+
<table>
<!-- TABLE BODY -->
<tbody>
<!-- TABLE HEADER -->
<th valign="bottom">Classes</th>
<th valign="bottom">Number of<br/>meanshapes</th>
<th valign="bottom">Masks</th>
<th valign="bottom">Paper<br/>reference</th>
<th valign="bottom">3D<br/>IoU</th>
<th valign="bottom">Download</th>
<!-- ROW: aeroplane 8 meanshapes -->
<tr>
<td align="center">Aeroplane</td>
<td align="center">8</td>
<td align="center">GT + MaskRCNN</td>
<td align="center">Table 2</td>
<td align="center">0.460</td>
<td align="center"><a href="https://ailb-web.ing.unimore.it/publicfiles/drive/3DV 2021 - mcmr/pretrained_weights/Table_2/aeroplane_8_meanshapes/net_latest.pth">model</a></td>
</tr>
<!-- ROW: car 10 meanshapes -->
<tr>
<td align="center">Car</td>
<td align="center">10</td>
<td align="center">GT + MaskRCNN</td>
<td align="center">Table 2</td>
<td align="center">0.684</td>
<td align="center"><a href="https://ailb-web.ing.unimore.it/publicfiles/drive/3DV 2021 - mcmr/pretrained_weights/Table_2/car_10_meanshapes/net_latest.pth">model</a></td>
</tr>
<!-- ROW: aeroplane+car 2 meanshapes -->
<tr>
<td align="center">Aeroplane,Car</td>
<td align="center">2</td>
<td align="center">GT + MaskRCNN</td>
<td align="center">Table 2</td>
<td align="center">0.567</td>
<td align="center"><a href="https://ailb-web.ing.unimore.it/publicfiles/drive/3DV 2021 - mcmr/pretrained_weights/Table_2/aereoplane_car_2_meanshapes/net_latest.pth">model</a></td>
</tr>
<!-- ROW: aeroplane+car 1 meanshape -->
<tr>
<td align="center">Aeroplane,Car</td>
<td align="center">1</td>
<td align="center">GT + PointRend</td>
<td align="center">Table 4</td>
<td align="center">0.532</td>
<td align="center"><a href="https://ailb-web.ing.unimore.it/publicfiles/drive/3DV 2021 - mcmr/pretrained_weights/Table_4/aeroplane_car_1_meanshape/net_latest.pth">model</a></td>
</tr>
<!-- ROW: aeroplane+car 2 meanshapes -->
<tr>
<td align="center">Aeroplane,Car</td>
<td align="center">2</td>
<td align="center">GT + PointRend</td>
<td align="center">Table 4</td>
<td align="center">0.552</td>
<td align="center"><a href="https://ailb-web.ing.unimore.it/publicfiles/drive/3DV 2021 - mcmr/pretrained_weights/Table_4/aeroplane_car_2_meanshapes/net_latest.pth">model</a></td>
</tr>
<!-- ROW: bicycle+bus+car+motorbike classes 1 meanshape -->
<tr>
<td align="center">Bicycle,Bus,Car,Motorbike</td>
<td align="center">1</td>
<td align="center">GT + PointRend</td>
<td align="center">Table 4</td>
<td align="center">0.517</td>
<td align="center"><a href="https://ailb-web.ing.unimore.it/publicfiles/drive/3DV 2021 - mcmr/pretrained_weights/Table_4/bicycle_bus_car_motorbike_1_meanshape/net_latest.pth">model</a></td>
</tr>
<!-- ROW: bicycle+bus+car+motorbike classes 4 meanshapes -->
<tr>
<td align="center">Bicycle,Bus,Car,Motorbike</td>
<td align="center">4</td>
<td align="center">GT + PointRend</td>
<td align="center">Table 4</td>
<td align="center">0.543</td>
<td align="center"><a href="https://ailb-web.ing.unimore.it/publicfiles/drive/3DV 2021 - mcmr/pretrained_weights/Table_4/bicycle_bus_car_motorbike_4_meanshapes/net_latest.pth">model</a></td>
</tr>
<!-- ROW: 12 Pascal3D+ classes 1 meanshape -->
<tr>
<td align="center">12 Pascal3D+ classes</td>
<td align="center">1</td>
<td align="center">GT + PointRend</td>
<td align="center">Table 4</td>
<td align="center">0.409</td>
<td align="center"><a href="https://ailb-web.ing.unimore.it/publicfiles/drive/3DV 2021 - mcmr/pretrained_weights/Table_4/12_pascal_classes_1_meanshape/net_latest.pth">model</a></td>
</tr>
<!-- ROW: 12 Pascal3D+ classes 12 meanshapes -->
<tr>
<td align="center">12 Pascal3D+ classes</td>
<td align="center">12</td>
<td align="center">GT + PointRend</td>
<td align="center">Table 4</td>
<td align="center">0.425</td>
<td align="center"><a href="https://ailb-web.ing.unimore.it/publicfiles/drive/3DV 2021 - mcmr/pretrained_weights/Table_4/12_pascal_classes_12_meanshapes/net_latest.pth">model</a></td>
</tr>
</tbody>
</table>

To compute the 3D IoU metric for Pascal3D+, click [here](https://ailb-web.ing.unimore.it/publicfiles/drive/3DV%202021%20-%20mcmr/drc-code-for-mcmr.zip) and download the MatLab code. We started from the [DRC](https://github.com/shubhtuls/drc) repo and adapt the code for reading our generated mat files containing the predicted meshes.

## CUB
<table>
<!-- TABLE BODY -->
<tbody>
<!-- TABLE HEADER -->
<th valign="bottom">Classes</th>
<th valign="bottom">Number of<br/>meanshapes</th>
<th valign="bottom">Masks</th>
<th valign="bottom">Paper<br/>reference</th>
<th valign="bottom">Mask<br/>IoU</th>
<th valign="bottom">Download</th>
<!-- ROW: bird 1 meanshape -->
<tr>
<td align="center">Bird</td>
<td align="center">1</td>
<td align="center">GT</td>
<td align="center">Table 3</td>
<td align="center">0.658</td>
<td align="center"><a href="https://ailb-web.ing.unimore.it/publicfiles/drive/3DV 2021 - mcmr/pretrained_weights/Table_3/CUB_1_meanshape/net_latest.pth">model</a></td>
</tr>
<!-- ROW: bird 14 meanshape -->
<tr>
<td align="center">Bird</td>
<td align="center">14</td>
<td align="center">GT</td>
<td align="center">Table 3</td>
<td align="center">0.642</td>
<td align="center"><a href="https://ailb-web.ing.unimore.it/publicfiles/drive/3DV 2021 - mcmr/pretrained_weights/Table_3/CUB_14_meanshapes/net_latest.pth">model</a></td>
</tr>
</tbody>
</table>
