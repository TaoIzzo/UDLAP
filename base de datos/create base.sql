CREATE TABLE CIUDAD (
    NombreCiudad VARCHAR(12),
    Pais VARCHAR(12),
    PRIMARY KEY (NombreCiudad , Pais)
);

create table LUGARAVISITAR(
Nombre VARCHAR(12), 
Ciudad VARCHAR(12), 
Pais VARCHAR(12), 
Direccion VARCHAR(12), 
Descripcion VARCHAR(40), 
Precio INTEGER NOT NULL, 
PRIMARY KEY(Nombre, Direccion)
);

create table FECHACIRCUITO(
Id VARCHAR(5), 
FechaSalida DATE, 
numPersonas INTEGER NOT NULL, 
PRIMARY KEY(FechaSalida)
);

create table CIRCUITO(
Id VARCHAR(5), 
Descripcion VARCHAR(20), 
CiudadSalida VARCHAR(12), 
PaisSalida VARCHAR(12), 
CiudadLlegada VARCHAR(12), 
PaisLlegada VARCHAR(12), 
Duracion INTEGER NOT NULL, 
PRIMARY KEY(Id)
);

create table ETAPA(
Id VARCHAR(5), 
Orden INTEGER NOT NULL, 
NombreLugar VARCHAR(12), 
Ciudad VARCHAR(12), 
Pais VARCHAR(12), 
Duracion INTEGER NOT NULL,
PRIMARY KEY(Orden, Duracion)
);

create table HOTEL(
Nombre VARCHAR(10), 
Ciudad VARCHAR(12), 
Pais VARCHAR(12), 
Direccion VARCHAR(20), 
PrecioCuarto FLOAT NOT NULL, 
PrecioDesayuno INTEGER NOT NULL, 
PRIMARY KEY(Nombre, Direccion, PrecioCuarto)
);

create table SIMULACION(
NombreCliente VARCHAR(20), 
FechaSalida DATE, 
FechaLlegada DATE, 
NumPersonas INTEGER, 
TipoReservacion VARCHAR(12), 
CostoViaje DOUBLE, 
PRIMARY KEY(NombreCliente, TipoReservacion, CostoViaje)
);

insert into CIUDAD values ('NEW YORK', 'USA');
insert into CIUDAD values ('CDMX', 'MEXICO');
insert into CIUDAD values ('LONDRES', 'REINO UNIDO');

insert into LUGARAVISITAR values ('TIMES SQUARE', 'NEW YORK', 'USA', 'Manhattan', 'ESCALERAS ROJAS DEL TKTS', 1200);
insert into LUGARAVISITAR values ('TORRE LAT', 'CDMX', 'MEXICO', 'Cuauhtémoc', 'MIRADOR', 800);
insert into LUGARAVISITAR values ('LONDON EYE', 'LONDRES', 'REINO UNIDO', 'County Hall', 'OJO DE LONDRES', 2100);

insert into FECHACIRCUITO values('NY891', '2021-01-12', 16);
insert into FECHACIRCUITO values('MX171', '2021-11-20', 24);
insert into FECHACIRCUITO values('UK001', '2022-02-11', 120);

insert into CIRCUITO values('NY891', 'Visitar NY', 'PUEBLA', 'MEXICO', 'NEW YORK', 'USA', 60);
insert into CIRCUITO values('MX171', 'Visitar CDMX', 'LONDRES', 'REINO UNIDO', 'CDMX', 'MEXICO', 30);
insert into CIRCUITO values('UK001', 'Visitar UK', 'NEW YORK', 'USA', 'LONDRES', 'REINO UNIDO', 15);

insert into ETAPA values('NY891', 8, 'TIMES SQUARE', 'NEW YORK', 'USA', 60);
insert into ETAPA values('MX171', 10, 'TORRE LAT', 'CDMX', 'MEXICO', 30);
insert into ETAPA values('UK001', 12, 'LONDON EYE', 'LONDRES', 'REINO UNIDO', 15);

insert into HOTEL values('Amazonas', 'CDMX', 'MEXICO', 'San Antonio', 482.00, 60);
insert into HOTEL values('William', 'NEW YORK', 'USA', 'Brooklyn NY', 6699.90, 240);
insert into HOTEL values('Savoy', 'LONDES', 'REINO UNIDO', 'Strand WC2R', 11200.00, 300);

insert into SIMULACION values('Alan', '2021-01-12', '2021-03-12', 16, 'Basica', 1200);
insert into SIMULACION values('Joseph', '2021-11-20', '2021-12-20', 24, 'Premium', 800);
insert into SIMULACION values('Monica', '2022-02-11', '2021-02-26', 120, 'Deluxe', 2100);