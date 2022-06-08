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
<td align="center"><a href="https://drive.google.com/file/d/1XZIpDJNyPQa3IFDaDiqFf8ClCSUK0Ck_/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: car 10 meanshapes -->
<tr>
<td align="center">Car</td>
<td align="center">10</td>
<td align="center">GT + MaskRCNN</td>
<td align="center">Table 2</td>
<td align="center">0.684</td>
<td align="center"><a href="https://drive.google.com/file/d/1r03Ci63J5FpyDxsGYOva_DSYqNw15MLQ/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: aeroplane+car 2 meanshapes -->
<tr>
<td align="center">Aeroplane,Car</td>
<td align="center">2</td>
<td align="center">GT + MaskRCNN</td>
<td align="center">Table 2</td>
<td align="center">0.567</td>
<td align="center"><a href="https://drive.google.com/file/d/1m2ff_wkPEh1hLN-hsBoXCMmcjxlbs6p2/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: aeroplane+car 1 meanshape -->
<tr>
<td align="center">Aeroplane,Car</td>
<td align="center">1</td>
<td align="center">GT + PointRend</td>
<td align="center">Table 4</td>
<td align="center">0.532</td>
<td align="center"><a href="https://drive.google.com/file/d/18xEQC500Yp_nIt-WHMYpx7mDbITYf9hd/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: aeroplane+car 2 meanshapes -->
<tr>
<td align="center">Aeroplane,Car</td>
<td align="center">2</td>
<td align="center">GT + PointRend</td>
<td align="center">Table 4</td>
<td align="center">0.552</td>
<td align="center"><a href="https://drive.google.com/file/d/14e4oi2nUfxSauRgG5BaNxPG_Rjwm_hkV/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: bicycle+bus+car+motorbike classes 1 meanshape -->
<tr>
<td align="center">Bicycle,Bus,Car,Motorbike</td>
<td align="center">1</td>
<td align="center">GT + PointRend</td>
<td align="center">Table 4</td>
<td align="center">0.517</td>
<td align="center"><a href="https://drive.google.com/file/d/1dAetRjglUnjbeCUbEa5uC0P8gorqPssN/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: bicycle+bus+car+motorbike classes 4 meanshapes -->
<tr>
<td align="center">Bicycle,Bus,Car,Motorbike</td>
<td align="center">4</td>
<td align="center">GT + PointRend</td>
<td align="center">Table 4</td>
<td align="center">0.543</td>
<td align="center"><a href="https://drive.google.com/file/d/1TFKrxbH_bCxvt75wIZqWIxrzTnRtxVdU/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: 12 Pascal3D+ classes 1 meanshape -->
<tr>
<td align="center">12 Pascal3D+ classes</td>
<td align="center">1</td>
<td align="center">GT + PointRend</td>
<td align="center">Table 4</td>
<td align="center">0.409</td>
<td align="center"><a href="https://drive.google.com/file/d/13XuQ4A7cunDZQ4w-GwLqzNyvttQxXhAY/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: 12 Pascal3D+ classes 12 meanshapes -->
<tr>
<td align="center">12 Pascal3D+ classes</td>
<td align="center">12</td>
<td align="center">GT + PointRend</td>
<td align="center">Table 4</td>
<td align="center">0.425</td>
<td align="center"><a href="https://drive.google.com/file/d/1AbAMASl62PJCNkG7VrpOOgyZI9JSeY2o/view?usp=sharing">model</a></td>
</tr>
</tbody>
</table>

To compute the 3D IoU metric for Pascal3D+, click [here](https://drive.google.com/file/d/1zrj-QtxfT2cX7N7YxljMgxoCnudjHahl/view?usp=sharing) and download the MatLab code. We started from the [DRC](https://github.com/shubhtuls/drc) repo and adapt the code for reading our generated mat files containing the predicted meshes.

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
<td align="center"><a href="https://drive.google.com/file/d/1Nz5sZS7kXWqX1A2_g3LyzQzsb8AoOiwz/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: bird 14 meanshape -->
<tr>
<td align="center">Bird</td>
<td align="center">14</td>
<td align="center">GT</td>
<td align="center">Table 3</td>
<td align="center">0.642</td>
<td align="center"><a href="https://drive.google.com/file/d/1PtKRzgGO7CrpIehBlWipj2SRPPBKTD10/view?usp=sharing">model</a></td>
</tr>
</tbody>
</table>